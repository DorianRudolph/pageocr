use std::{
    cell::Cell,
    collections::BTreeSet,
    env, fs,
    io::{self, IsTerminal, Read, Write},
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo};
use image::{GrayImage, RgbImage, imageops, imageops::FilterType};
use indicatif::{ProgressBar, ProgressStyle};
use llama_cpp_4::{
    context::LlamaContext,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaChatMessage, LlamaModel, Special, params::LlamaModelParams},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText},
    sampling::LlamaSampler,
};
use llama_cpp_sys_4 as llama_sys;
use log::debug;
use minijinja::{Environment, context};
use pathdiff::diff_paths;
use pdfium_auto::{bind_pdfium_from_path, ensure_pdfium_library, is_pdfium_cached};
use pdfium_render::prelude::*;
use regex::Regex;
use serde::Serialize;

const DEFAULT_MARGIN: u32 = 20;
const DEFAULT_LONG_EDGE: u32 = 1540;
const DEFAULT_THRESHOLD: u8 = 245;
const DEFAULT_RENDER_DPI: u32 = 200;
const DEFAULT_DETECT_DPI: u32 = 72;
const LIGHTON_BBOX_PADDING: u32 = 5;
const HELP_EXAMPLES: &str = "\
Examples:
  pageocr scan.pdf
  pageocr --page-range 3-5 scan.pdf
  pageocr page1.png page2.png > out.md
  pageocr --variant bbox --extract-images-dir imgs paper.pdf
  cat page.png | pageocr";

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum LightonVariant {
    Default,
    Bbox,
    BboxSoup,
}

#[derive(Debug, Clone, Copy)]
struct LightonModelSpec {
    variant: LightonVariant,
    repo_id: &'static str,
    model_filename: &'static str,
    mmproj_filename: &'static str,
    supports_bbox_exports: bool,
}

impl LightonVariant {
    fn spec(self) -> LightonModelSpec {
        match self {
            Self::Default => LightonModelSpec {
                variant: self,
                repo_id: "Mungert/LightOnOCR-2-1B-GGUF",
                model_filename: "LightOnOCR-2-1B-bf16.gguf",
                mmproj_filename: "LightOnOCR-2-1B-bf16.mmproj",
                supports_bbox_exports: false,
            },
            Self::Bbox => LightonModelSpec {
                variant: self,
                repo_id: "noctrex/LightOnOCR-2-1B-bbox-GGUF",
                model_filename: "LightOnOCR-2-1B-bbox-BF16.gguf",
                mmproj_filename: "mmproj-BF16.gguf",
                supports_bbox_exports: true,
            },
            Self::BboxSoup => LightonModelSpec {
                variant: self,
                repo_id: "noctrex/LightOnOCR-2-1B-bbox-soup-GGUF",
                model_filename: "LightOnOCR-2-1B-bbox-soup-BF16.gguf",
                mmproj_filename: "mmproj-BF16.gguf",
                supports_bbox_exports: true,
            },
        }
    }

    fn display_name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Bbox => "bbox",
            Self::BboxSoup => "bbox-soup",
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "OCR PDFs and images with LightOnOCR.",
    after_help = HELP_EXAMPLES
)]
struct Args {
    #[arg(
        long = "variant",
        alias = "lighton-variant",
        value_enum,
        default_value_t = LightonVariant::Default,
        help_heading = "General"
    )]
    lighton_variant: LightonVariant,

    #[arg(long, help_heading = "Advanced")]
    model: Option<PathBuf>,

    #[arg(long, help_heading = "Advanced")]
    mmproj: Option<PathBuf>,

    #[arg(
        short = 'p',
        long = "page-range",
        alias = "pdf-pages",
        help_heading = "General"
    )]
    pdf_pages: Option<String>,

    #[arg(long, default_value_t = DEFAULT_MARGIN, help_heading = "General")]
    margin: u32,

    #[arg(
        long = "max-edge",
        alias = "long-edge",
        default_value_t = DEFAULT_LONG_EDGE,
        help_heading = "General"
    )]
    long_edge: u32,

    #[arg(long, default_value_t = DEFAULT_THRESHOLD, help_heading = "Advanced")]
    threshold: u8,

    #[arg(long, default_value_t = DEFAULT_RENDER_DPI, help_heading = "Advanced")]
    render_dpi: u32,

    #[arg(long, default_value_t = DEFAULT_DETECT_DPI, help_heading = "Advanced")]
    detect_dpi: u32,

    /// Enable PDF auto-cropping before OCR.
    #[arg(long, default_value_t = false, help_heading = "General")]
    autocrop: bool,

    #[arg(
        long = "dump-rendered-pages-dir",
        alias = "dump-crops-dir",
        help_heading = "Advanced"
    )]
    dump_crops_dir: Option<PathBuf>,

    /// Export bbox-detected image regions and rewrite markdown image links.
    #[arg(
        long = "extract-images-dir",
        alias = "export-detected-images-dir",
        help_heading = "General"
    )]
    export_detected_images_dir: Option<PathBuf>,

    #[arg(short = 'o', long, help_heading = "General")]
    output: Option<PathBuf>,

    #[arg(short = 't', long = "template", alias = "output-template", help_heading = "General")]
    output_template: Option<PathBuf>,

    /// Show per-page OCR preview while processing PDFs.
    #[arg(long, default_value_t = false, help_heading = "General")]
    live_preview: bool,

    #[arg(long, default_value_t = false, help_heading = "Advanced")]
    show_llama_logs: bool,

    /// Extra user text after the image. Empty matches the server example.
    #[arg(long, default_value = "", help_heading = "Advanced")]
    prompt: String,

    #[arg(long, default_value_t = 2048, help_heading = "Advanced")]
    max_tokens: usize,

    #[arg(long, default_value_t = 4096, help_heading = "Advanced")]
    n_ctx: u32,

    #[arg(long, default_value_t = 512, help_heading = "Advanced")]
    n_batch: u32,

    #[arg(short = 'j', long, help_heading = "Advanced")]
    n_threads: Option<i32>,

    #[arg(long, default_value_t = 0.2, help_heading = "Advanced")]
    temperature: f32,

    #[arg(long, default_value_t = 0, help_heading = "Advanced")]
    top_k: i32,

    #[arg(long, default_value_t = 0.9, help_heading = "Advanced")]
    top_p: f32,

    #[arg(long, default_value_t = 1234, help_heading = "Advanced")]
    seed: u32,

    #[arg(long, help_heading = "Advanced")]
    no_gpu: bool,

    #[arg(long, help_heading = "Advanced")]
    print_prompt: bool,

    #[arg(value_name = "INPUT")]
    inputs: Vec<PathBuf>,
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

struct OcrRuntime {
    backend: LlamaBackend,
    model: LlamaModel,
    mtmd_ctx: MtmdContext,
    media_marker: String,
}

#[derive(Debug, Clone, Serialize)]
struct OutputPage {
    page_number: usize,
    markdown: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BboxDetection {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
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

fn main() -> Result<()> {
    init_logging();
    let args = Args::parse();
    validate_args(&args)?;
    let lighton_spec = args.lighton_variant.spec();

    let (model_path, mmproj_path) = resolve_lighton_paths(&args)?;
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

    debug!("model  : {}", model_path.display());
    debug!("mmproj : {}", mmproj_path.display());
    debug!("variant: {}", lighton_spec.variant.display_name());
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
                let markdown = postprocess_lighton_markdown(
                    runtime.ocr_rgb_image(&image, &mut lctx, &args)?,
                    &image,
                    completed_pages,
                    &args,
                    export_detected_images_dir.as_deref(),
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
                    let markdown = postprocess_lighton_markdown(
                        runtime
                            .ocr_rgb_image(&image, &mut lctx, &args)
                            .with_context(|| format!("OCR failed on PDF page {page_number}"))?,
                        &image,
                        completed_pages,
                        &args,
                        export_detected_images_dir.as_deref(),
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

    Ok(())
}

impl OcrRuntime {
    fn new(args: &Args, model_path: &Path, mmproj_path: &Path) -> Result<Self> {
        let mut backend = LlamaBackend::init()?;
        if !args.show_llama_logs {
            backend.void_logs();
            silence_mtmd_logs();
        }

        let model_params = LlamaModelParams::default().with_n_gpu_layers(99);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .with_context(|| format!("failed to load model from {}", model_path.display()))?;

        let chat_template = model
            .get_chat_template(64 * 1024)
            .context("failed to read tokenizer.chat_template from the GGUF")?;
        let media_marker = detect_media_marker(&chat_template);

        let mmproj_params = MtmdContextParams::default()
            .use_gpu(!args.no_gpu)
            .n_threads(resolved_n_threads(args.n_threads))
            .print_timings(false)
            .media_marker(Some(&media_marker))?;

        let mtmd_ctx = MtmdContext::init_from_file(mmproj_path, &model, mmproj_params)
            .with_context(|| format!("failed to load mmproj from {}", mmproj_path.display()))?;
        if !args.show_llama_logs {
            silence_mtmd_logs();
        }

        if !mtmd_ctx.supports_vision() {
            bail!(
                "the mmproj at {} does not expose vision support",
                mmproj_path.display()
            );
        }

        Ok(Self {
            backend,
            model,
            mtmd_ctx,
            media_marker,
        })
    }

    fn ocr_rgb_image(
        &self,
        image: &RgbImage,
        lctx: &mut LlamaContext<'_>,
        args: &Args,
    ) -> Result<String> {
        let bitmap = MtmdBitmap::from_rgb(image.width(), image.height(), image.as_raw())
            .context("failed to create bitmap from RGB image")?;
        self.ocr_bitmap(&bitmap, lctx, args)
    }

    fn ocr_bitmap(
        &self,
        bitmap: &MtmdBitmap,
        lctx: &mut LlamaContext<'_>,
        args: &Args,
    ) -> Result<String> {
        let formatted_prompt = self.format_prompt(&args.prompt)?;
        if args.print_prompt {
            debug!("formatted prompt:\n{formatted_prompt}");
        }

        lctx.clear_kv_cache();

        let input_text = MtmdInputText::new(&formatted_prompt, true, true);
        let bitmap_refs = [bitmap];
        let mut chunks = MtmdInputChunks::new();
        self.mtmd_ctx
            .tokenize(&input_text, &bitmap_refs, &mut chunks)
            .context("failed to tokenize the multimodal prompt")?;

        let mut n_past = 0i32;
        self.mtmd_ctx
            .eval_chunks(
                lctx.as_ptr(),
                &chunks,
                0,
                0,
                args.n_batch as i32,
                true,
                &mut n_past,
            )
            .context("failed to evaluate prompt chunks")?;

        let mut sampler = build_sampler(args);
        let mut batch = LlamaBatch::new(1, 1);
        let mut response = String::new();

        for _ in 0..args.max_tokens {
            let token = sampler.sample(&lctx, -1);
            if self.model.is_eog_token(token) || token == self.model.token_eos() {
                break;
            }

            let piece = self
                .model
                .token_to_str(token, Special::Tokenize)
                .unwrap_or_default();
            response.push_str(&piece);

            batch.clear();
            batch.add(token, n_past, &[0], true)?;
            lctx.decode(&mut batch)?;
            sampler.accept(token);

            n_past += 1;
        }

        Ok(response)
    }

    fn new_context(&self, args: &Args) -> Result<LlamaContext<'_>> {
        let n_threads = resolved_n_threads(args.n_threads);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(args.n_ctx))
            .with_n_batch(args.n_batch)
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);
        Ok(self.model.new_context(&self.backend, ctx_params)?)
    }

    fn format_prompt(&self, prompt: &str) -> Result<String> {
        let user_content = build_user_content(&self.media_marker, prompt);
        let chat = vec![LlamaChatMessage::new("user".to_owned(), user_content)?];
        self.model
            .apply_chat_template(None, chat, true)
            .context("failed to apply the GGUF chat template")
    }
}

impl PdfProgressUi {
    fn new(total_pages: usize, show_llama_logs: bool, live_preview: bool) -> Result<Self> {
        if !should_use_fancy_progress(show_llama_logs, io::stderr().is_terminal()) {
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

fn should_use_fancy_progress(show_llama_logs: bool, stderr_is_terminal: bool) -> bool {
    stderr_is_terminal && !show_llama_logs
}

fn resolved_n_threads(configured: Option<i32>) -> i32 {
    configured.filter(|&count| count > 0).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|count| count.get() as i32)
            .unwrap_or(1)
    })
}

fn validate_args(args: &Args) -> Result<()> {
    if args.long_edge == 0 {
        bail!("--long-edge must be greater than 0");
    }
    if args.long_edge > DEFAULT_LONG_EDGE {
        bail!(
            "--long-edge must be at most {DEFAULT_LONG_EDGE} for LightOnOCR, got {}",
            args.long_edge
        );
    }
    if args.margin.saturating_mul(2) >= args.long_edge {
        bail!("--margin must be less than half of --long-edge");
    }
    if args.render_dpi == 0 {
        bail!("--render-dpi must be greater than 0");
    }
    if args.detect_dpi == 0 {
        bail!("--detect-dpi must be greater than 0");
    }
    if args.export_detected_images_dir.is_some()
        && !args.lighton_variant.spec().supports_bbox_exports
    {
        bail!("--export-detected-images-dir requires --lighton-variant bbox or bbox-soup");
    }
    Ok(())
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

fn detect_inputs(args: &Args) -> Result<Vec<DetectedInput>> {
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
    args: &Args,
    detected_inputs: Vec<DetectedInput>,
    pdfium: Option<&Pdfium>,
) -> Result<Vec<PreparedInput>> {
    if args.pdf_pages.is_some()
        && (detected_inputs.len() != 1
            || !matches!(detected_inputs.first(), Some(DetectedInput::Pdf { .. })))
    {
        bail!("--pdf-pages can only be used with a single PDF input");
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
        OutputStrategy::StreamDefaultStdout => progress_ui
            .write_output_page_to_stdout(output_page, completed_pages, total_output_pages),
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

fn resolve_lighton_paths(args: &Args) -> Result<(PathBuf, PathBuf)> {
    let spec = args.lighton_variant.spec();
    let cache = lighton_cache();
    let default_paths = resolve_default_lighton_paths(
        &cache,
        spec,
        args.model.is_none(),
        args.mmproj.is_none(),
        prompt_for_lighton_download,
        |spec, missing| download_lighton_files(&cache, spec, missing),
    )?;

    let model_path = args
        .model
        .as_deref()
        .map(normalize_path)
        .or(default_paths.model)
        .expect("default LightOn model path must be resolved");
    let mmproj_path = args
        .mmproj
        .as_deref()
        .map(normalize_path)
        .or(default_paths.mmproj)
        .expect("default LightOn mmproj path must be resolved");

    Ok((model_path, mmproj_path))
}

#[derive(Debug, Default)]
struct DefaultLightonPaths {
    model: Option<PathBuf>,
    mmproj: Option<PathBuf>,
}

fn resolve_default_lighton_paths<FPrompt, FDownload>(
    cache: &Cache,
    spec: LightonModelSpec,
    need_model: bool,
    need_mmproj: bool,
    mut prompt_for_download: FPrompt,
    mut download_missing: FDownload,
) -> Result<DefaultLightonPaths>
where
    FPrompt: FnMut(LightonModelSpec, &[&'static str]) -> Result<bool>,
    FDownload: FnMut(LightonModelSpec, &[&'static str]) -> Result<Vec<(&'static str, PathBuf)>>,
{
    let mut paths = DefaultLightonPaths::default();
    let mut missing = Vec::new();

    if need_model {
        if let Some(path) = cached_lighton_path(cache, spec, spec.model_filename) {
            paths.model = Some(normalize_path(&path));
        } else {
            missing.push(spec.model_filename);
        }
    }

    if need_mmproj {
        if let Some(path) = cached_lighton_path(cache, spec, spec.mmproj_filename) {
            paths.mmproj = Some(normalize_path(&path));
        } else {
            missing.push(spec.mmproj_filename);
        }
    }

    if missing.is_empty() {
        return Ok(paths);
    }

    if !prompt_for_download(spec, &missing)? {
        bail!("{}", declined_lighton_download_message(spec, &missing));
    }

    for (filename, path) in download_missing(spec, &missing)? {
        let normalized = normalize_path(&path);
        if filename == spec.model_filename {
            paths.model = Some(normalized);
        } else if filename == spec.mmproj_filename {
            paths.mmproj = Some(normalized);
        }
    }

    if need_model && paths.model.is_none() {
        bail!("download did not produce {}", spec.model_filename);
    }
    if need_mmproj && paths.mmproj.is_none() {
        bail!("download did not produce {}", spec.mmproj_filename);
    }

    Ok(paths)
}

fn cached_lighton_path(
    cache: &Cache,
    spec: LightonModelSpec,
    filename: &'static str,
) -> Option<PathBuf> {
    cache
        .repo(lighton_repo(spec))
        .get(filename)
        .filter(|path| path.exists())
}

fn lighton_cache() -> Cache {
    env_path("HF_HUB_CACHE")
        .or_else(|| env_path("HUGGINGFACE_HUB_CACHE"))
        .map(Cache::new)
        .unwrap_or_else(Cache::from_env)
}

fn lighton_repo(spec: LightonModelSpec) -> Repo {
    Repo::model(spec.repo_id.to_owned())
}

fn env_path(key: &str) -> Option<PathBuf> {
    env::var_os(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn prompt_for_lighton_download(spec: LightonModelSpec, missing: &[&'static str]) -> Result<bool> {
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        bail!("{}", noninteractive_lighton_download_message(spec, missing));
    }

    eprintln!(
        "LightOn {} model files are not cached:",
        spec.variant.display_name()
    );
    for filename in missing {
        eprintln!("  - {filename}");
    }
    eprint!("Download the missing file(s) from Hugging Face now? [y/N] ");
    io::stderr().flush().context("failed to flush prompt")?;

    let mut response = String::new();
    io::stdin()
        .read_line(&mut response)
        .context("failed to read download confirmation")?;

    Ok(matches!(
        response.trim().to_ascii_lowercase().as_str(),
        "y" | "yes"
    ))
}

fn download_lighton_files(
    cache: &Cache,
    spec: LightonModelSpec,
    missing: &[&'static str],
) -> Result<Vec<(&'static str, PathBuf)>> {
    let api = ApiBuilder::from_cache(cache.clone())
        .with_progress(true)
        .build()
        .context("failed to initialize Hugging Face API client")?;
    let repo = api.repo(lighton_repo(spec));

    missing
        .iter()
        .map(|filename| {
            repo.download(filename)
                .with_context(|| format!("failed to download {filename}"))
                .map(|path| (*filename, path))
        })
        .collect()
}

fn noninteractive_lighton_download_message(
    spec: LightonModelSpec,
    missing: &[&'static str],
) -> String {
    format!(
        "LightOn {} model file(s) are missing from the Hugging Face cache: {}. Rerun in an interactive terminal to approve the download, or prefetch them with `hf download {} {}`.",
        spec.variant.display_name(),
        missing.join(", "),
        spec.repo_id,
        missing.join(" ")
    )
}

fn declined_lighton_download_message(spec: LightonModelSpec, missing: &[&'static str]) -> String {
    format!(
        "LightOn {} model download declined; required file(s) are missing from the Hugging Face cache: {}.",
        spec.variant.display_name(),
        missing.join(", ")
    )
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

fn silence_mtmd_logs() {
    unsafe extern "C" fn noop(
        _level: llama_sys::ggml_log_level,
        _text: *const ::std::os::raw::c_char,
        _user_data: *mut ::std::os::raw::c_void,
    ) {
    }

    unsafe {
        llama_sys::mtmd_log_set(Some(noop), std::ptr::null_mut());
        llama_sys::mtmd_helper_log_set(Some(noop), std::ptr::null_mut());
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

fn detect_media_marker(chat_template: &str) -> String {
    for marker in [
        "<|image_pad|>",
        "<|video_pad|>",
        "<image>",
        MtmdContext::default_marker(),
    ] {
        if chat_template.contains(marker) {
            return marker.to_owned();
        }
    }

    MtmdContext::default_marker().to_owned()
}

fn build_user_content(media_marker: &str, prompt: &str) -> String {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        media_marker.to_owned()
    } else {
        format!("{media_marker}\n{prompt}")
    }
}

fn build_sampler(args: &Args) -> LlamaSampler {
    if args.temperature <= 0.0 {
        return LlamaSampler::greedy();
    }

    let mut samplers = Vec::new();
    if args.top_k > 0 {
        samplers.push(LlamaSampler::top_k(args.top_k));
    }
    if args.top_p > 0.0 && args.top_p < 1.0 {
        samplers.push(LlamaSampler::top_p(args.top_p, 1));
    }
    samplers.push(LlamaSampler::temp(args.temperature));
    samplers.push(LlamaSampler::dist(args.seed));

    LlamaSampler::chain_simple(samplers)
}

fn postprocess_lighton_markdown(
    markdown: String,
    source_image: &RgbImage,
    page_number: usize,
    args: &Args,
    export_dir: Option<&Path>,
) -> Result<String> {
    if !args.lighton_variant.spec().supports_bbox_exports {
        return Ok(markdown);
    }

    let Some(export_dir) = export_dir else {
        return Ok(markdown);
    };

    export_bbox_detections(
        &markdown,
        source_image,
        page_number,
        export_dir,
        args.output.as_deref(),
    )
}

fn export_bbox_detections(
    markdown: &str,
    source_image: &RgbImage,
    page_number: usize,
    export_dir: &Path,
    output_path: Option<&Path>,
) -> Result<String> {
    let pattern = bbox_detection_pattern();
    let mut rewritten = String::with_capacity(markdown.len());
    let mut last_end = 0usize;

    for (detection_index, captures) in pattern.captures_iter(markdown).enumerate() {
        let matched = captures.get(0).expect("bbox match must exist");
        rewritten.push_str(&markdown[last_end..matched.start()]);

        let detection = parse_bbox_detection(&captures);
        let filename = format!("page_{page_number:04}_image_{:04}.png", detection_index + 1);
        let export_path = export_dir.join(&filename);

        match crop_bbox_detection(source_image, detection) {
            Ok(crop) => {
                crop.save(&export_path).with_context(|| {
                    format!(
                        "failed to save detected image crop to {}",
                        export_path.display()
                    )
                })?;
                let link_path = markdown_export_path(&export_path, output_path)?;
                rewritten.push_str(&format!("![image]({link_path})"));
            }
            Err(err) => {
                debug!(
                    "skipping bbox detection on page {} due to invalid crop: {err:#}",
                    page_number
                );
                rewritten.push_str(matched.as_str());
            }
        }

        last_end = matched.end();
    }

    rewritten.push_str(&markdown[last_end..]);
    Ok(rewritten)
}

fn bbox_detection_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(r"!\[[^\]]*\]\((image_\d+\.png)\)\s*(\d+),(\d+),(\d+),(\d+)")
            .expect("bbox detection regex must compile")
    })
}

fn parse_bbox_detection(captures: &regex::Captures<'_>) -> BboxDetection {
    BboxDetection {
        left: captures[2].parse().expect("bbox left must be numeric"),
        top: captures[3].parse().expect("bbox top must be numeric"),
        right: captures[4].parse().expect("bbox right must be numeric"),
        bottom: captures[5].parse().expect("bbox bottom must be numeric"),
    }
}

fn crop_bbox_detection(source_image: &RgbImage, detection: BboxDetection) -> Result<RgbImage> {
    let width = source_image.width();
    let height = source_image.height();
    let left = normalized_bbox_to_pixels(detection.left, width);
    let top = normalized_bbox_to_pixels(detection.top, height);
    let right = normalized_bbox_to_pixels(detection.right, width);
    let bottom = normalized_bbox_to_pixels(detection.bottom, height);

    if right <= left || bottom <= top {
        bail!(
            "empty bbox crop after normalization: {},{},{},{}",
            detection.left,
            detection.top,
            detection.right,
            detection.bottom
        );
    }

    let padded_left = left.saturating_sub(LIGHTON_BBOX_PADDING);
    let padded_top = top.saturating_sub(LIGHTON_BBOX_PADDING);
    let padded_right = right.saturating_add(LIGHTON_BBOX_PADDING).min(width);
    let padded_bottom = bottom.saturating_add(LIGHTON_BBOX_PADDING).min(height);

    if padded_right <= padded_left || padded_bottom <= padded_top {
        bail!("bbox crop became empty after padding");
    }

    Ok(imageops::crop_imm(
        source_image,
        padded_left,
        padded_top,
        padded_right - padded_left,
        padded_bottom - padded_top,
    )
    .to_image())
}

fn normalized_bbox_to_pixels(value: u32, size: u32) -> u32 {
    let clamped = value.min(1000) as u64;
    ((clamped * size as u64) / 1000) as u32
}

fn markdown_export_path(exported_path: &Path, output_path: Option<&Path>) -> Result<String> {
    let base_dir = match output_path {
        Some(path) => absolute_path(path)?
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| anyhow::anyhow!("output path has no parent: {}", path.display()))?,
        None => env::current_dir().context("failed to read current working directory")?,
    };

    let display_path = diff_paths(exported_path, &base_dir).unwrap_or_else(|| exported_path.into());
    Ok(display_path.to_string_lossy().replace('\\', "/"))
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

fn render_pdf_page(page: &PdfPage, args: &Args) -> Result<RgbImage> {
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

    let available_edge = args.long_edge - (args.margin * 2);
    let scale = available_edge as f32 / content_width.max(content_height);
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
    use std::cell::Cell;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_path(filename: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(filename)
    }

    fn test_args() -> Args {
        Args {
            lighton_variant: LightonVariant::Default,
            model: None,
            mmproj: None,
            pdf_pages: None,
            margin: DEFAULT_MARGIN,
            long_edge: DEFAULT_LONG_EDGE,
            threshold: DEFAULT_THRESHOLD,
            render_dpi: DEFAULT_RENDER_DPI,
            detect_dpi: DEFAULT_DETECT_DPI,
            autocrop: false,
            dump_crops_dir: None,
            export_detected_images_dir: None,
            output: None,
            output_template: None,
            live_preview: false,
            show_llama_logs: false,
            prompt: String::new(),
            max_tokens: 256,
            n_ctx: 4096,
            n_batch: 256,
            n_threads: None,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.9,
            seed: 1234,
            no_gpu: true,
            print_prompt: false,
            inputs: Vec::new(),
        }
    }

    #[test]
    fn args_default_to_autocrop_disabled() {
        let args = Args::try_parse_from(["pageocr", "input.pdf"]).unwrap();
        assert!(!args.autocrop);
    }

    #[test]
    fn args_accept_autocrop_flag() {
        let args = Args::try_parse_from(["pageocr", "input.pdf", "--autocrop"]).unwrap();
        assert!(args.autocrop);
    }

    #[test]
    fn args_default_to_live_preview_disabled() {
        let args = Args::try_parse_from(["pageocr", "input.pdf"]).unwrap();
        assert!(!args.live_preview);
    }

    #[test]
    fn args_accept_live_preview_flag() {
        let args = Args::try_parse_from(["pageocr", "input.pdf", "--live-preview"]).unwrap();
        assert!(args.live_preview);
    }

    #[test]
    fn args_default_to_default_lighton_variant() {
        let args = Args::try_parse_from(["pageocr", "input.pdf"]).unwrap();
        assert_eq!(args.lighton_variant, LightonVariant::Default);
    }

    #[test]
    fn args_accept_bbox_lighton_variants() {
        let bbox =
            Args::try_parse_from(["pageocr", "input.pdf", "--variant", "bbox"]).unwrap();
        assert_eq!(bbox.lighton_variant, LightonVariant::Bbox);

        let bbox_soup =
            Args::try_parse_from(["pageocr", "input.pdf", "--variant", "bbox-soup"])
                .unwrap();
        assert_eq!(bbox_soup.lighton_variant, LightonVariant::BboxSoup);
    }

    #[test]
    fn args_keep_legacy_flag_aliases() {
        let args = Args::try_parse_from([
            "pageocr",
            "input.pdf",
            "--lighton-variant",
            "bbox",
            "--pdf-pages",
            "1-2",
            "--long-edge",
            "1200",
            "--output-template",
            "template.j2",
            "--export-detected-images-dir",
            "imgs",
            "--dump-crops-dir",
            "debug",
        ])
        .unwrap();

        assert_eq!(args.lighton_variant, LightonVariant::Bbox);
        assert_eq!(args.pdf_pages.as_deref(), Some("1-2"));
        assert_eq!(args.long_edge, 1200);
        assert_eq!(args.output_template.as_deref(), Some(Path::new("template.j2")));
        assert_eq!(
            args.export_detected_images_dir.as_deref(),
            Some(Path::new("imgs"))
        );
        assert_eq!(args.dump_crops_dir.as_deref(), Some(Path::new("debug")));
    }

    #[test]
    fn args_accept_multiple_inputs() {
        let args = Args::try_parse_from(["pageocr", "page1.png", "page2.png"]).unwrap();
        assert_eq!(
            args.inputs,
            vec![PathBuf::from("page1.png"), PathBuf::from("page2.png")]
        );
    }

    #[test]
    fn args_accept_explicit_n_threads() {
        let args = Args::try_parse_from(["pageocr", "-j", "7", "input.pdf"]).unwrap();
        assert_eq!(args.n_threads, Some(7));
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
        assert!(should_use_fancy_progress(false, true));
        assert!(!should_use_fancy_progress(true, true));
        assert!(!should_use_fancy_progress(false, false));
    }

    #[test]
    fn resolved_n_threads_prefers_positive_override() {
        assert_eq!(resolved_n_threads(Some(7)), 7);
    }

    #[test]
    fn resolved_n_threads_auto_detects_for_missing_or_non_positive_values() {
        assert!(resolved_n_threads(None) >= 1);
        assert!(resolved_n_threads(Some(0)) >= 1);
        assert!(resolved_n_threads(Some(-1)) >= 1);
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
                .contains("--pdf-pages can only be used with a single PDF input"),
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

        let bounds = detect_content_bounds(&image, DEFAULT_THRESHOLD);

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
    fn cached_lighton_path_uses_hf_hub_cache_repo() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let cache_root = env::temp_dir().join(format!("ocr-cli-{unique}"));
        let cache = Cache::new(cache_root.join("hub"));
        let spec = LightonVariant::Bbox.spec();
        let repo = cache.repo(lighton_repo(spec));
        let resolved = repo.pointer_path("123abc").join(spec.model_filename);

        fs::create_dir_all(resolved.parent().unwrap()).unwrap();
        repo.create_ref("123abc").unwrap();
        fs::write(&resolved, "").unwrap();

        let actual = cached_lighton_path(&cache, spec, spec.model_filename);

        assert_eq!(actual, Some(resolved.clone()));

        fs::remove_dir_all(cache_root).unwrap();
    }

    #[test]
    fn resolve_default_lighton_paths_uses_downloaded_files_after_confirmation() {
        let cache = Cache::new(env::temp_dir().join("ocr-cli-tests-unused-cache"));
        let prompt_called = Cell::new(false);

        let paths = resolve_default_lighton_paths(
            &cache,
            LightonVariant::BboxSoup.spec(),
            true,
            true,
            |spec, missing| {
                prompt_called.set(true);
                assert_eq!(spec.variant, LightonVariant::BboxSoup);
                assert_eq!(
                    missing,
                    &["LightOnOCR-2-1B-bbox-soup-BF16.gguf", "mmproj-BF16.gguf"]
                );
                Ok(true)
            },
            |_, missing| {
                Ok(missing
                    .iter()
                    .map(|filename| {
                        (
                            *filename,
                            PathBuf::from("/tmp").join(format!("downloaded-{filename}")),
                        )
                    })
                    .collect())
            },
        )
        .unwrap();

        assert!(prompt_called.get());
        assert_eq!(
            paths.model,
            Some(PathBuf::from("/tmp").join("downloaded-LightOnOCR-2-1B-bbox-soup-BF16.gguf"))
        );
        assert_eq!(
            paths.mmproj,
            Some(PathBuf::from("/tmp").join("downloaded-mmproj-BF16.gguf"))
        );
    }

    #[test]
    fn resolve_default_lighton_paths_errors_when_download_is_declined() {
        let cache = Cache::new(env::temp_dir().join("ocr-cli-tests-unused-cache"));

        let err = resolve_default_lighton_paths(
            &cache,
            LightonVariant::Bbox.spec(),
            true,
            false,
            |_, _| Ok(false),
            |_, _| unreachable!("download should not be attempted when declined"),
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("download declined"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    #[ignore = "requires cached LightOn model files and runs full OCR"]
    fn lighton_ocr_smoke_test_on_fixture_image() -> Result<()> {
        let cache = lighton_cache();
        let spec = LightonVariant::Default.spec();
        let Some(model_path) = cached_lighton_path(&cache, spec, spec.model_filename) else {
            return Ok(());
        };
        let Some(mmproj_path) = cached_lighton_path(&cache, spec, spec.mmproj_filename) else {
            return Ok(());
        };

        let args = test_args();
        let runtime = OcrRuntime::new(&args, &model_path, &mmproj_path)?;
        let mut lctx = runtime.new_context(&args)?;
        let image = load_image(&ImageSource::Path(fixture_path(
            "noaa_eating_lake_michigan_fish_page1.png",
        )))?;
        let markdown = runtime.ocr_rgb_image(&image, &mut lctx, &args)?;
        let normalized = markdown.to_ascii_lowercase();

        assert!(
            normalized.contains("lake michigan")
                || normalized.contains("what are pcbs")
                || normalized.contains("pcbs"),
            "unexpected OCR output: {markdown}"
        );

        Ok(())
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

    #[test]
    fn export_detected_images_requires_bbox_variant() {
        let mut args = test_args();
        args.export_detected_images_dir = Some(PathBuf::from("images"));

        let err = validate_args(&args).unwrap_err();

        assert!(
            err.to_string().contains(
                "--export-detected-images-dir requires --lighton-variant bbox or bbox-soup"
            ),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn bbox_exports_are_renamed_per_page_and_rewritten_relative_to_output() -> Result<()> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!("ocr-cli-bbox-test-{unique}"));
        let export_dir = root.join("images");
        let output_dir = root.join("output");
        fs::create_dir_all(&export_dir)?;
        fs::create_dir_all(&output_dir)?;

        let mut image = RgbImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let pixel = if x < 50 {
                    image::Rgb([255, 0, 0])
                } else {
                    image::Rgb([0, 255, 0])
                };
                image.put_pixel(x, y, pixel);
            }
        }

        let markdown = concat!(
            "before\n",
            "![image](image_1.png) 0,0,500,1000\n",
            "middle\n",
            "![image](image_2.png) 500,0,1000,1000\n",
            "after\n"
        );
        let rewritten = export_bbox_detections(
            markdown,
            &image,
            3,
            &export_dir,
            Some(&output_dir.join("result.md")),
        )?;

        assert!(rewritten.contains("![image](../images/page_0003_image_0001.png)"));
        assert!(rewritten.contains("![image](../images/page_0003_image_0002.png)"));
        assert!(!rewritten.contains("image_1.png) 0,0,500,1000"));
        assert!(!rewritten.contains("image_2.png) 500,0,1000,1000"));

        let left = image::open(export_dir.join("page_0003_image_0001.png"))?.to_rgb8();
        let right = image::open(export_dir.join("page_0003_image_0002.png"))?.to_rgb8();
        assert_eq!(left.width(), 55);
        assert_eq!(right.width(), 55);
        assert_eq!(left.height(), 100);
        assert_eq!(right.height(), 100);
        assert_eq!(left.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
        assert_eq!(
            right.get_pixel(right.width() - 1, 0),
            &image::Rgb([0, 255, 0])
        );

        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn bbox_exports_accept_compact_lighton_markdown_format() -> Result<()> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!("ocr-cli-bbox-compact-test-{unique}"));
        let export_dir = root.join("images");
        fs::create_dir_all(&export_dir)?;

        let image = RgbImage::from_pixel(100, 100, image::Rgb([12, 34, 56]));
        let rewritten = export_bbox_detections(
            "![image](image_1.png)57,50,220,170",
            &image,
            1,
            &export_dir,
            None,
        )?;

        assert!(rewritten.ends_with("/images/page_0001_image_0001.png)"));
        assert!(export_dir.join("page_0001_image_0001.png").exists());

        fs::remove_dir_all(root)?;
        Ok(())
    }
}
