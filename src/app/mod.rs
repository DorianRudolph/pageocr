mod bbox;
mod cli;
mod model;
mod reasoning;
mod runtime;

use std::{
    cell::Cell,
    collections::BTreeSet,
    env, fs,
    io::{self, IsTerminal, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use image::{GrayImage, RgbImage, imageops, imageops::FilterType};
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;
use minijinja::{Environment, context};
use pdfium_auto::{bind_pdfium_from_path, ensure_pdfium_library, is_pdfium_cached};
use pdfium_render::prelude::*;
use serde::Serialize;

use self::{
    bbox::postprocess_markdown,
    cli::ResolvedArgs,
    model::ModelFamily,
    reasoning::{ReasoningTraceEntry, extract_reasoning_trace, render_reasoning_json},
    runtime::OcrRuntime,
};

#[cfg(test)]
use self::cli::Args;

pub fn run() -> Result<()> {
    init_logging();
    let args = ResolvedArgs::parse()?;

    let (model_path, mmproj_path) = args.selection.resolve_paths(
        args.model_override.as_deref(),
        args.mmproj_override.as_deref(),
    )?;
    ensure_exists(&model_path, "model")?;
    ensure_exists(&mmproj_path, "mmproj")?;

    let dump_crops_dir = args.dump_crops_dir.as_deref().map(normalize_path);
    if let Some(path) = dump_crops_dir.as_deref() {
        ensure_dir(path, "dump_crops_dir")?;
    }
    let export_detected_images_dir = args
        .export_detected_images_dir
        .as_deref()
        .map(absolute_path)
        .transpose()?;
    if let Some(path) = export_detected_images_dir.as_deref() {
        ensure_dir(path, "export_detected_images_dir")?;
    }

    let detected_inputs = detect_inputs(&args)?;
    let pdfium = detected_inputs
        .iter()
        .any(|input| matches!(input, DetectedInput::Pdf { .. }))
        .then(bind_pdfium)
        .transpose()?;
    let prepared_inputs = prepare_inputs(&args, detected_inputs, pdfium.as_ref())?;
    let total_output_pages = prepared_inputs
        .iter()
        .map(PreparedInput::page_count)
        .sum::<usize>();
    if total_output_pages == 0 {
        bail!("no pages selected for OCR");
    }

    let output_strategy = output_strategy(args.output_template.as_deref(), args.output.as_deref());

    let runtime = OcrRuntime::new(&args, &model_path, &mmproj_path)?;
    let mut lctx = runtime.new_context(&args)?;
    let progress_ui =
        PdfProgressUi::new(total_output_pages, args.show_llama_logs, args.live_preview)?;
    let mut rendered_pages = match output_strategy {
        OutputStrategy::StreamDefaultStdout => None,
        OutputStrategy::RenderAtEnd => Some(Vec::with_capacity(total_output_pages)),
    };
    let mut completed_pages = 0usize;
    let mut reasoning_entries = Vec::new();

    debug!("model  : {}", model_path.display());
    debug!("mmproj : {}", mmproj_path.display());
    debug!(
        "family : {} ({})",
        args.selection.family().display_name(),
        args.selection.display_name()
    );
    debug!("marker : {}", runtime.media_marker);
    if let Some(dir) = dump_crops_dir.as_deref() {
        debug!("crops  : {}", dir.display());
    }
    if let Some(dir) = export_detected_images_dir.as_deref() {
        debug!("images : {}", dir.display());
    }

    for input in &prepared_inputs {
        match input {
            PreparedInput::Image { source } => {
                progress_ui.start_page(completed_pages + 1, completed_pages, total_output_pages);
                let image = downscale_to_long_edge(load_image(source)?, args.long_edge);
                if let Some(dir) = dump_crops_dir.as_deref() {
                    let dump_name = format!("page_{:04}.png", completed_pages + 1);
                    let dump_path = dump_rgb_image(&image, dir, &dump_name)?;
                    debug!("crop image: {}", dump_path.display());
                }
                completed_pages += 1;
                let extracted = extract_reasoning_trace(
                    runtime.ocr_rgb_image(&image, &mut lctx, &args)?,
                    completed_pages,
                );
                reasoning_entries.extend(extracted.entries);
                let markdown = postprocess_markdown(
                    extracted.markdown,
                    &image,
                    completed_pages,
                    args.selection.supports_bbox_exports(),
                    export_detected_images_dir.as_deref(),
                    args.output.as_deref(),
                )?;
                let output_page = OutputPage {
                    page_number: completed_pages,
                    markdown,
                };
                progress_ui.finish_page(&output_page, completed_pages, total_output_pages);
                emit_output_page(
                    &progress_ui,
                    &mut rendered_pages,
                    &output_page,
                    completed_pages,
                    total_output_pages,
                    output_strategy,
                )?;
            }
            PreparedInput::Pdf {
                source,
                page_numbers,
            } => {
                let document =
                    load_pdf_source(pdfium.as_ref().expect("pdfium must be available"), source)
                        .context("failed to open PDF input")?;
                for page_number in page_numbers {
                    progress_ui.start_page(
                        completed_pages + 1,
                        completed_pages,
                        total_output_pages,
                    );
                    let page = document
                        .pages()
                        .get((*page_number - 1) as u16)
                        .with_context(|| format!("failed to load PDF page {page_number}"))?;
                    let image = render_pdf_page(&page, &args)
                        .with_context(|| format!("failed to render PDF page {page_number}"))?;
                    if let Some(dir) = dump_crops_dir.as_deref() {
                        let dump_name = format!("page_{:04}.png", completed_pages + 1);
                        let dump_path = dump_rgb_image(&image, dir, &dump_name)?;
                        debug!("crop image: {}", dump_path.display());
                    }
                    completed_pages += 1;
                    let extracted = extract_reasoning_trace(
                        runtime
                            .ocr_rgb_image(&image, &mut lctx, &args)
                            .with_context(|| format!("OCR failed on PDF page {page_number}"))?,
                        completed_pages,
                    );
                    reasoning_entries.extend(extracted.entries);
                    let markdown = postprocess_markdown(
                        extracted.markdown,
                        &image,
                        completed_pages,
                        args.selection.supports_bbox_exports(),
                        export_detected_images_dir.as_deref(),
                        args.output.as_deref(),
                    )?;
                    let output_page = OutputPage {
                        page_number: completed_pages,
                        markdown,
                    };
                    progress_ui.finish_page(&output_page, completed_pages, total_output_pages);
                    emit_output_page(
                        &progress_ui,
                        &mut rendered_pages,
                        &output_page,
                        completed_pages,
                        total_output_pages,
                        output_strategy,
                    )?;
                }
            }
        }
    }
    progress_ui.finish();

    if let Some(pages) = rendered_pages.as_deref() {
        let rendered = render_output_document(pages, args.output_template.as_deref())?;
        write_output(&rendered, args.output.as_deref())?;
    }
    if let Some(path) = args.reasoning_json.as_deref() {
        write_reasoning_output(&reasoning_entries, path)?;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct CropBounds {
    left: f32,
    bottom: f32,
    right: f32,
    top: f32,
}

impl CropBounds {
    fn content_width(self, page_width: f32) -> f32 {
        page_width - self.left - self.right
    }

    fn content_height(self, page_height: f32) -> f32 {
        page_height - self.top - self.bottom
    }
}

#[derive(Debug, Clone, Serialize)]
struct OutputPage {
    page_number: usize,
    markdown: String,
}

enum PdfProgressUi {
    Fancy {
        progress: ProgressBar,
        live_preview: bool,
    },
    Quiet {
        live_preview: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputStrategy {
    StreamDefaultStdout,
    RenderAtEnd,
}

#[derive(Debug, Clone)]
enum DetectedInput {
    Image { source: ImageSource },
    Pdf { source: PdfSource },
}

#[derive(Debug, Clone)]
enum PreparedInput {
    Image {
        source: ImageSource,
    },
    Pdf {
        source: PdfSource,
        page_numbers: Vec<usize>,
    },
}

#[derive(Debug, Clone)]
enum ImageSource {
    Path(PathBuf),
    Bytes { bytes: Vec<u8> },
}

#[derive(Debug, Clone)]
enum PdfSource {
    Path(PathBuf),
    Bytes { bytes: Vec<u8> },
}

impl PdfProgressUi {
    fn new(total_pages: usize, show_llama_logs: bool, live_preview: bool) -> Result<Self> {
        if !should_use_fancy_progress(total_pages, show_llama_logs, io::stderr().is_terminal()) {
            return Ok(Self::Quiet { live_preview });
        }

        let progress = ProgressBar::new(total_pages as u64);
        let style = ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}",
        )
        .context("failed to configure progress bar style")?
        .progress_chars("##-");
        progress.set_style(style);
        progress.set_position(0);
        progress.set_message("starting");
        Ok(Self::Fancy {
            progress,
            live_preview,
        })
    }

    fn start_page(&self, page_number: usize, completed_pages: usize, total_pages: usize) {
        let message = format!(
            "decoding page {page_number} ({}/{total_pages})",
            completed_pages + 1
        );
        match self {
            Self::Fancy { progress, .. } => progress.set_message(message),
            Self::Quiet { .. } => {}
        }
    }

    fn finish_page(&self, page: &OutputPage, completed_pages: usize, total_pages: usize) {
        match self {
            Self::Fancy {
                progress,
                live_preview,
            } => {
                progress.set_position(completed_pages as u64);
                if *live_preview {
                    progress.println(render_live_page_preview(page, completed_pages, total_pages));
                }
            }
            Self::Quiet { live_preview } => {
                if *live_preview {
                    eprintln!(
                        "{}",
                        render_live_page_preview(page, completed_pages, total_pages)
                    );
                }
            }
        }
    }

    fn finish(&self) {
        if let Self::Fancy { progress, .. } = self {
            progress.finish_and_clear();
        }
    }

    fn write_output_page_to_stdout(
        &self,
        page: &OutputPage,
        completed_pages: usize,
        total_pages: usize,
    ) -> Result<()> {
        match self {
            Self::Fancy { progress, .. } => progress.suspend(|| {
                write_default_output_page_to_stdout(page, completed_pages, total_pages)
            }),
            Self::Quiet { .. } => {
                write_default_output_page_to_stdout(page, completed_pages, total_pages)
            }
        }
    }
}

fn should_use_fancy_progress(
    total_pages: usize,
    show_llama_logs: bool,
    stderr_is_terminal: bool,
) -> bool {
    total_pages > 1 && stderr_is_terminal && !show_llama_logs
}

fn init_logging() {
    let env = env_logger::Env::default().default_filter_or("warn");
    let mut builder = env_logger::Builder::from_env(env);
    builder.format_timestamp(None);
    let _ = builder.try_init();
}

fn normalize_path(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(env::current_dir()
            .context("failed to read current working directory")?
            .join(path))
    }
}

fn ensure_exists(path: &Path, label: &str) -> Result<()> {
    if path.exists() {
        Ok(())
    } else {
        bail!("{label} path does not exist: {}", path.display())
    }
}

fn ensure_dir(path: &Path, label: &str) -> Result<()> {
    if path.exists() && !path.is_dir() {
        bail!("{label} must be a directory path: {}", path.display());
    }
    fs::create_dir_all(path)
        .with_context(|| format!("failed to create directory {}", path.display()))
}

fn output_strategy(template_path: Option<&Path>, output_path: Option<&Path>) -> OutputStrategy {
    if template_path.is_none() && output_path.is_none() {
        OutputStrategy::StreamDefaultStdout
    } else {
        OutputStrategy::RenderAtEnd
    }
}

impl PreparedInput {
    fn page_count(&self) -> usize {
        match self {
            Self::Image { .. } => 1,
            Self::Pdf { page_numbers, .. } => page_numbers.len(),
        }
    }
}

fn detect_inputs(args: &ResolvedArgs) -> Result<Vec<DetectedInput>> {
    if args.inputs.is_empty() {
        if io::stdin().is_terminal() {
            bail!("pass one or more input files, or pipe a PDF/image into stdin");
        }
        return Ok(vec![detect_input_from_stdin(read_stdin_bytes()?)?]);
    }

    let mut used_stdin = false;
    let mut inputs = Vec::with_capacity(args.inputs.len());
    for input in &args.inputs {
        if input == Path::new("-") {
            if used_stdin {
                bail!("stdin can only be used once");
            }
            used_stdin = true;
            inputs.push(detect_input_from_stdin(read_stdin_bytes()?)?);
        } else {
            inputs.push(detect_input_from_path(normalize_path(input))?);
        }
    }
    Ok(inputs)
}

fn prepare_inputs(
    args: &ResolvedArgs,
    detected_inputs: Vec<DetectedInput>,
    pdfium: Option<&Pdfium>,
) -> Result<Vec<PreparedInput>> {
    if args.pdf_pages.is_some()
        && (detected_inputs.len() != 1
            || !matches!(detected_inputs.first(), Some(DetectedInput::Pdf { .. })))
    {
        bail!("--pages can only be used with a single PDF input");
    }

    let mut prepared = Vec::with_capacity(detected_inputs.len());
    for input in detected_inputs {
        match input {
            DetectedInput::Image { source } => prepared.push(PreparedInput::Image { source }),
            DetectedInput::Pdf { source } => {
                let page_numbers = {
                    let document = load_pdf_source(
                        pdfium.expect("pdfium must be available for PDF inputs"),
                        &source,
                    )
                    .context("failed to open PDF input")?;
                    parse_page_range(args.pdf_pages.as_deref(), document.pages().len() as usize)?
                };
                prepared.push(PreparedInput::Pdf {
                    source,
                    page_numbers,
                });
            }
        }
    }

    Ok(prepared)
}

fn emit_output_page(
    progress_ui: &PdfProgressUi,
    rendered_pages: &mut Option<Vec<OutputPage>>,
    output_page: &OutputPage,
    completed_pages: usize,
    total_output_pages: usize,
    output_strategy: OutputStrategy,
) -> Result<()> {
    match output_strategy {
        OutputStrategy::StreamDefaultStdout => progress_ui.write_output_page_to_stdout(
            output_page,
            completed_pages,
            total_output_pages,
        ),
        OutputStrategy::RenderAtEnd => {
            rendered_pages
                .as_mut()
                .expect("rendered page buffer must exist")
                .push(output_page.clone());
            Ok(())
        }
    }
}

fn detect_input_from_path(path: PathBuf) -> Result<DetectedInput> {
    ensure_exists(&path, "input")?;
    let mut file = fs::File::open(&path)
        .with_context(|| format!("failed to open input {}", path.display()))?;
    let mut header = [0u8; 32];
    let bytes_read = file
        .read(&mut header)
        .with_context(|| format!("failed to read input {}", path.display()))?;
    detect_input_kind(&header[..bytes_read]).map(|is_pdf| {
        if is_pdf {
            DetectedInput::Pdf {
                source: PdfSource::Path(path),
            }
        } else {
            DetectedInput::Image {
                source: ImageSource::Path(path),
            }
        }
    })
}

fn detect_input_from_stdin(bytes: Vec<u8>) -> Result<DetectedInput> {
    detect_input_kind(&bytes).map(|is_pdf| {
        if is_pdf {
            DetectedInput::Pdf {
                source: PdfSource::Bytes { bytes },
            }
        } else {
            DetectedInput::Image {
                source: ImageSource::Bytes { bytes },
            }
        }
    })
}

fn detect_input_kind(bytes: &[u8]) -> Result<bool> {
    if bytes.is_empty() {
        bail!("input was empty");
    }
    if bytes.starts_with(b"%PDF-") {
        return Ok(true);
    }
    if image::guess_format(bytes).is_ok() {
        return Ok(false);
    }
    bail!("could not detect whether the input is a PDF or image")
}

fn read_stdin_bytes() -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    io::stdin()
        .lock()
        .read_to_end(&mut bytes)
        .context("failed to read input from stdin")?;
    if bytes.is_empty() {
        bail!("stdin was empty");
    }
    Ok(bytes)
}

fn load_image(source: &ImageSource) -> Result<RgbImage> {
    let image = match source {
        ImageSource::Path(path) => image::open(path)
            .with_context(|| format!("failed to open image {}", path.display()))
            .map(|image| image.to_rgb8()),
        ImageSource::Bytes { bytes } => image::load_from_memory(bytes)
            .context("failed to decode image from stdin")
            .map(|image| image.to_rgb8()),
    }?;

    Ok(image)
}

fn load_pdf_source<'a>(pdfium: &'a Pdfium, source: &'a PdfSource) -> Result<PdfDocument<'a>> {
    match source {
        PdfSource::Path(path) => pdfium
            .load_pdf_from_file(path, None)
            .with_context(|| format!("failed to open PDF {}", path.display())),
        PdfSource::Bytes { bytes } => pdfium
            .load_pdf_from_byte_slice(bytes, None)
            .context("failed to open PDF from stdin"),
    }
}

fn render_output_document(pages: &[OutputPage], template_path: Option<&Path>) -> Result<String> {
    match template_path {
        Some(path) => fs::read_to_string(path)
            .with_context(|| format!("failed to read output template {}", path.display()))
            .and_then(|template_source| {
                render_output_document_from_source(pages, &template_source)
            }),
        None => Ok(render_default_output_document(pages)),
    }
}

fn render_output_document_from_source(
    pages: &[OutputPage],
    template_source: &str,
) -> Result<String> {
    let mut env = Environment::new();
    env.add_template("output", template_source)
        .context("failed to parse output template")?;
    let template = env
        .get_template("output")
        .context("failed to load output template")?;
    template
        .render(context! {
            pages => pages,
            page_count => pages.len(),
            is_multipage => pages.len() > 1,
        })
        .context("failed to render output template")
}

fn render_default_output_document(pages: &[OutputPage]) -> String {
    pages
        .iter()
        .enumerate()
        .map(|(index, page)| render_default_output_page(page, index + 1, pages.len()))
        .collect()
}

fn render_default_output_page(
    page: &OutputPage,
    completed_pages: usize,
    total_pages: usize,
) -> String {
    let markdown = page.markdown.trim_end_matches('\n');
    if total_pages <= 1 {
        markdown.to_owned()
    } else if completed_pages < total_pages {
        format!(
            "<!-- Page {} -->\n\n{markdown}\n\n---\n\n",
            page.page_number
        )
    } else {
        format!("<!-- Page {} -->\n\n{markdown}", page.page_number)
    }
}

fn render_live_page_preview(
    page: &OutputPage,
    completed_pages: usize,
    total_pages: usize,
) -> String {
    let markdown = page.markdown.trim_end_matches('\n');
    if total_pages <= 1 {
        markdown.to_owned()
    } else if completed_pages < total_pages {
        format!("<!-- Page {} -->\n\n{markdown}\n\n---\n", page.page_number)
    } else {
        format!("<!-- Page {} -->\n\n{markdown}", page.page_number)
    }
}

fn write_default_output_page_to_stdout(
    page: &OutputPage,
    completed_pages: usize,
    total_pages: usize,
) -> Result<()> {
    let fragment = render_default_output_page(page, completed_pages, total_pages);
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(fragment.as_bytes())
        .context("failed to write output to stdout")?;
    if completed_pages == total_pages && !fragment.ends_with('\n') {
        stdout
            .write_all(b"\n")
            .context("failed to finish stdout output")?;
    }
    stdout.flush().context("failed to flush stdout output")
}

fn write_output(rendered: &str, output_path: Option<&Path>) -> Result<()> {
    if let Some(path) = output_path {
        fs::write(path, rendered)
            .with_context(|| format!("failed to write output to {}", path.display()))?;
        debug!("output : {}", path.display());
    } else {
        print!("{rendered}");
        if !rendered.ends_with('\n') {
            println!();
        }
    }

    Ok(())
}

fn write_reasoning_output(entries: &[ReasoningTraceEntry], output_path: &Path) -> Result<()> {
    let rendered = render_reasoning_json(entries)?;
    fs::write(output_path, rendered).with_context(|| {
        format!(
            "failed to write reasoning JSON output to {}",
            output_path.display()
        )
    })
}

fn dump_rgb_image(image: &RgbImage, dir: &Path, filename: &str) -> Result<PathBuf> {
    let output_path = dir.join(filename);
    image
        .save(&output_path)
        .with_context(|| format!("failed to save debug crop to {}", output_path.display()))?;
    Ok(output_path)
}

fn bind_pdfium() -> Result<Pdfium> {
    let was_cached = is_pdfium_cached();
    let last_percent = Cell::new(None::<u64>);
    let progress = |downloaded: u64, total: Option<u64>| {
        let Some(total) = total.filter(|total| *total > 0) else {
            return;
        };

        let percent = (downloaded.saturating_mul(100) / total).min(100);
        if last_percent.get() == Some(percent) || (percent < 100 && percent % 10 != 0) {
            return;
        }

        last_percent.set(Some(percent));
        eprintln!("downloading Pdfium runtime library... {percent}%");
    };

    let path = ensure_pdfium_library(if was_cached { None } else { Some(&progress) }).context(
        "failed to locate or download Pdfium; set PDFIUM_LIB_PATH to an existing library to skip runtime download",
    )?;

    bind_pdfium_from_path(&path)
        .with_context(|| format!("failed to bind Pdfium from {}", path.display()))
}

fn parse_page_range(spec: Option<&str>, total_pages: usize) -> Result<Vec<usize>> {
    if total_pages == 0 {
        bail!("the PDF has no pages");
    }

    let Some(spec) = spec.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok((1..=total_pages).collect());
    };

    let mut pages = BTreeSet::new();
    for segment in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((start, end)) = segment.split_once('-') {
            let start = if start.trim().is_empty() {
                1
            } else {
                start
                    .trim()
                    .parse::<usize>()
                    .with_context(|| format!("invalid page number in range {segment:?}"))?
            };
            let end = if end.trim().is_empty() {
                total_pages
            } else {
                end.trim()
                    .parse::<usize>()
                    .with_context(|| format!("invalid page number in range {segment:?}"))?
            };

            if start == 0 || end == 0 || start > end {
                bail!("invalid page range {segment:?}");
            }

            for page in start..=end {
                pages.insert(page);
            }
        } else {
            let page = segment
                .parse::<usize>()
                .with_context(|| format!("invalid page number {segment:?}"))?;
            if page == 0 {
                bail!("page numbers are 1-based, got 0");
            }
            pages.insert(page);
        }
    }

    let pages = pages.into_iter().collect::<Vec<_>>();
    if let Some(page) = pages.iter().copied().find(|page| *page > total_pages) {
        bail!("page {page} is out of range for a PDF with {total_pages} page(s)");
    }

    if pages.is_empty() {
        bail!("page range did not select any pages");
    }

    Ok(pages)
}

fn render_pdf_page(page: &PdfPage, args: &ResolvedArgs) -> Result<RgbImage> {
    let crop = if args.autocrop {
        detect_crop_bounds(page, args.threshold, args.detect_dpi)?
    } else {
        full_page_crop_bounds()
    };
    let page_width = page.width().value;
    let page_height = page.height().value;
    let content_width = crop.content_width(page_width);
    let content_height = crop.content_height(page_height);

    if content_width <= 0.0 || content_height <= 0.0 {
        bail!("auto-crop produced an empty page");
    }

    let scale = pdf_render_scale(args, content_width, content_height);
    let target_width = (page_width * scale).ceil().max(1.0) as i32;
    let target_height = (page_height * scale).ceil().max(1.0) as i32;

    let render_config = PdfRenderConfig::new()
        .set_target_width(target_width)
        .set_target_height(target_height);
    let rendered = page
        .render_with_config(&render_config)
        .context("pdfium page render failed")?
        .as_image()
        .to_rgb8();

    let margin_units = args.margin as f32 / scale;
    let expanded_left = (crop.left - margin_units).max(0.0);
    let expanded_right = (crop.right - margin_units).max(0.0);
    let expanded_top = (crop.top - margin_units).max(0.0);
    let expanded_bottom = (crop.bottom - margin_units).max(0.0);

    let left = (expanded_left * scale).floor().max(0.0) as u32;
    let top = (expanded_top * scale).floor().max(0.0) as u32;
    let right = ((page_width - expanded_right) * scale)
        .ceil()
        .min(rendered.width() as f32) as u32;
    let bottom = ((page_height - expanded_bottom) * scale)
        .ceil()
        .min(rendered.height() as f32) as u32;

    if right <= left || bottom <= top {
        bail!("cropped page bounds were empty after expansion");
    }

    let cropped = imageops::crop_imm(&rendered, left, top, right - left, bottom - top).to_image();
    Ok(downscale_to_long_edge(cropped, args.long_edge))
}

fn pdf_render_scale(args: &ResolvedArgs, content_width: f32, content_height: f32) -> f32 {
    match args.selection.family() {
        ModelFamily::Lighton => {
            let available_edge = args.long_edge - (args.margin * 2);
            available_edge as f32 / content_width.max(content_height)
        }
        ModelFamily::Qianfan => args.render_dpi as f32 / 72.0,
    }
}

fn downscale_to_long_edge(image: RgbImage, long_edge: u32) -> RgbImage {
    let max_edge = image.width().max(image.height());
    if max_edge <= long_edge {
        return image;
    }

    let resize_scale = long_edge as f32 / max_edge as f32;
    let width = ((image.width() as f32) * resize_scale).round().max(1.0) as u32;
    let height = ((image.height() as f32) * resize_scale).round().max(1.0) as u32;
    imageops::resize(&image, width, height, FilterType::Lanczos3)
}

fn full_page_crop_bounds() -> CropBounds {
    CropBounds {
        left: 0.0,
        bottom: 0.0,
        right: 0.0,
        top: 0.0,
    }
}

fn detect_crop_bounds(page: &PdfPage, threshold: u8, detect_dpi: u32) -> Result<CropBounds> {
    let scale = detect_dpi as f32 / 72.0;
    let target_width = (page.width().value * scale).ceil().max(1.0) as i32;
    let target_height = (page.height().value * scale).ceil().max(1.0) as i32;
    let detect_config = PdfRenderConfig::new()
        .set_target_width(target_width)
        .set_target_height(target_height);
    let image = page
        .render_with_config(&detect_config)
        .context("low-resolution crop render failed")?
        .as_image()
        .to_luma8();

    let Some((left, top, right, bottom)) = detect_content_bounds(&image, threshold) else {
        return Ok(CropBounds {
            left: 0.0,
            bottom: 0.0,
            right: 0.0,
            top: 0.0,
        });
    };

    Ok(CropBounds {
        left: left as f32 / scale,
        bottom: (image.height() - bottom) as f32 / scale,
        right: (image.width() - right) as f32 / scale,
        top: top as f32 / scale,
    })
}

fn detect_content_bounds(image: &GrayImage, threshold: u8) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = image.width();
    let mut min_y = image.height();
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[0] < threshold {
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x + 1);
            max_y = max_y.max(y + 1);
        }
    }

    found.then_some((min_x, min_y, max_x, max_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::path::PathBuf;

    fn fixture_path(filename: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(filename)
    }

    fn test_args() -> ResolvedArgs {
        ResolvedArgs::from_args(Args::try_parse_from(["pageocr", "input.pdf"]).unwrap()).unwrap()
    }

    #[test]
    fn default_output_streams_only_for_plain_stdout() {
        assert_eq!(
            output_strategy(None, None),
            OutputStrategy::StreamDefaultStdout
        );
        assert_eq!(
            output_strategy(Some(Path::new("template.j2")), None),
            OutputStrategy::RenderAtEnd
        );
        assert_eq!(
            output_strategy(None, Some(Path::new("output.md"))),
            OutputStrategy::RenderAtEnd
        );
    }

    #[test]
    fn fancy_progress_requires_tty_and_hidden_llama_logs() {
        assert!(should_use_fancy_progress(2, false, true));
        assert!(!should_use_fancy_progress(2, true, true));
        assert!(!should_use_fancy_progress(2, false, false));
    }

    #[test]
    fn fancy_progress_is_disabled_for_single_page_runs() {
        assert!(!should_use_fancy_progress(1, false, true));
    }

    #[test]
    fn image_downscale_respects_long_edge_without_upscaling() {
        let large = RgbImage::new(3000, 1500);
        let resized = downscale_to_long_edge(large, 1540);
        assert_eq!((resized.width(), resized.height()), (1540, 770));

        let small = RgbImage::new(600, 400);
        let unchanged = downscale_to_long_edge(small, 1540);
        assert_eq!((unchanged.width(), unchanged.height()), (600, 400));
    }

    #[test]
    fn lighton_pdf_render_scale_fits_content_to_long_edge() {
        let args = test_args();
        let scale = pdf_render_scale(&args, 612.0, 792.0);

        assert!((scale - (1500.0 / 792.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn qianfan_pdf_render_scale_uses_render_dpi() {
        let args = ResolvedArgs::from_args(
            Args::try_parse_from(["pageocr", "--qianfan", "input.pdf"]).unwrap(),
        )
        .unwrap();
        let scale = pdf_render_scale(&args, 300.0, 500.0);

        assert!((scale - (200.0 / 72.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn pdf_pages_requires_a_single_pdf_input() {
        let mut args = test_args();
        args.pdf_pages = Some("1-2".to_owned());

        let err = prepare_inputs(
            &args,
            vec![
                DetectedInput::Pdf {
                    source: PdfSource::Bytes {
                        bytes: b"%PDF-1.4".to_vec(),
                    },
                },
                DetectedInput::Image {
                    source: ImageSource::Bytes {
                        bytes: vec![0x89, b'P', b'N', b'G'],
                    },
                },
            ],
            None,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("--pages can only be used with a single PDF input"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn openstax_fixture_pdf_has_expected_page_count() -> Result<()> {
        let pdfium = bind_pdfium()?;
        let fixture = fixture_path("openstax_university_physics_selected_pages.pdf");
        let document = pdfium.load_pdf_from_file(&fixture, None)?;
        assert_eq!(document.pages().len(), 3);
        Ok(())
    }

    #[test]
    fn fixture_image_has_expected_detected_content_bounds() {
        let image = image::open(fixture_path("noaa_eating_lake_michigan_fish_page1.png"))
            .unwrap()
            .to_luma8();

        let bounds = detect_content_bounds(&image, cli::DEFAULT_THRESHOLD);

        assert_eq!(bounds, Some((78, 25, 1199, 1495)));
    }

    #[test]
    fn math_textbook_fixture_image_has_expected_dimensions() {
        let image = image::open(fixture_path("calculus_made_easy_page272.png"))
            .unwrap()
            .to_luma8();
        assert_eq!((image.width(), image.height()), (701, 1028));
    }

    #[test]
    fn default_output_renderer_keeps_single_page_plain() {
        let rendered = render_default_output_document(&[OutputPage {
            page_number: 1,
            markdown: "single page markdown".to_owned(),
        }]);

        assert_eq!(rendered, "single page markdown");
    }

    #[test]
    fn render_output_document_without_template_uses_default_renderer() {
        let pages = vec![OutputPage {
            page_number: 1,
            markdown: "single page markdown".to_owned(),
        }];

        let rendered = render_output_document(&pages, None).unwrap();

        assert_eq!(rendered, render_default_output_document(&pages));
    }

    #[test]
    fn default_output_renderer_formats_multiple_pages() {
        let rendered = render_default_output_document(&[
            OutputPage {
                page_number: 1,
                markdown: "page one".to_owned(),
            },
            OutputPage {
                page_number: 3,
                markdown: "page three".to_owned(),
            },
        ]);

        assert_eq!(
            rendered,
            "<!-- Page 1 -->\n\npage one\n\n---\n\n<!-- Page 3 -->\n\npage three"
        );
    }

    #[test]
    fn custom_output_template_still_formats_multiple_pages() {
        let rendered = render_output_document_from_source(
            &[OutputPage {
                page_number: 1,
                markdown: "page one".to_owned(),
            },
            OutputPage {
                page_number: 3,
                markdown: "page three".to_owned(),
            }],
            "{% for page in pages %}{{ page.page_number }}={{ page.markdown }}{% if not loop.last %}|{% endif %}{% endfor %}",
        )
        .unwrap();

        assert_eq!(rendered, "1=page one|3=page three");
    }
}
