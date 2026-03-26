use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;

use super::model::{LightonVariant, ModelFamily, ModelSelection, QianfanModel, ReasoningMode};

pub const DEFAULT_MARGIN: u32 = 20;
pub const DEFAULT_THRESHOLD: u8 = 245;
pub const DEFAULT_RENDER_DPI: u32 = 200;
pub const DEFAULT_DETECT_DPI: u32 = 72;

const HELP_EXAMPLES: &str = "\
Examples:
  pageocr scan.pdf
  pageocr --page-range 3-5 scan.pdf
  pageocr page1.png page2.png > out.md
  pageocr --variant bbox --extract-images-dir imgs paper.pdf
  pageocr --family qianfan --qianfan-model q8 page.png
  cat page.png | pageocr";

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "OCR PDFs and images with local multimodal OCR models.",
    after_help = HELP_EXAMPLES
)]
pub struct Args {
    #[arg(
        long = "family",
        alias = "model-family",
        value_enum,
        default_value_t = ModelFamily::Lighton,
        help_heading = "General"
    )]
    model_family: ModelFamily,

    #[arg(
        long = "variant",
        alias = "lighton-variant",
        value_enum,
        default_value_t = LightonVariant::Default,
        help_heading = "General"
    )]
    lighton_variant: LightonVariant,

    #[arg(
        long = "qianfan-model",
        alias = "qianfan-quant",
        value_enum,
        default_value_t = QianfanModel::Q8,
        help_heading = "General"
    )]
    qianfan_model: QianfanModel,

    #[arg(
        long,
        value_enum,
        default_value_t = ReasoningMode::Off,
        help_heading = "General"
    )]
    reasoning: ReasoningMode,

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
        help_heading = "General",
        value_name = "PIXELS"
    )]
    long_edge: Option<u32>,

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

    #[arg(
        short = 't',
        long = "template",
        alias = "output-template",
        help_heading = "General"
    )]
    output_template: Option<PathBuf>,

    /// Show per-page OCR preview while processing PDFs.
    #[arg(long, default_value_t = false, help_heading = "General")]
    live_preview: bool,

    #[arg(long, default_value_t = false, help_heading = "Advanced")]
    show_llama_logs: bool,

    /// Extra user text after the image. Defaults depend on the selected model family.
    #[arg(long, help_heading = "Advanced")]
    prompt: Option<String>,

    #[arg(long, help_heading = "Advanced")]
    max_tokens: Option<usize>,

    #[arg(long, help_heading = "Advanced")]
    n_ctx: Option<u32>,

    #[arg(long, help_heading = "Advanced")]
    n_batch: Option<u32>,

    #[arg(short = 'j', long, help_heading = "Advanced")]
    n_threads: Option<i32>,

    #[arg(long, help_heading = "Advanced")]
    temperature: Option<f32>,

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

#[derive(Debug, Clone)]
pub struct ResolvedArgs {
    pub selection: ModelSelection,
    pub model_override: Option<PathBuf>,
    pub mmproj_override: Option<PathBuf>,
    pub pdf_pages: Option<String>,
    pub margin: u32,
    pub long_edge: u32,
    pub threshold: u8,
    pub render_dpi: u32,
    pub detect_dpi: u32,
    pub autocrop: bool,
    pub dump_crops_dir: Option<PathBuf>,
    pub export_detected_images_dir: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub output_template: Option<PathBuf>,
    pub live_preview: bool,
    pub show_llama_logs: bool,
    pub prompt: String,
    pub max_tokens: usize,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: Option<i32>,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
    pub no_gpu: bool,
    pub print_prompt: bool,
    pub inputs: Vec<PathBuf>,
}

impl ResolvedArgs {
    pub fn parse() -> Result<Self> {
        Self::from_args(Args::parse())
    }

    pub fn from_args(args: Args) -> Result<Self> {
        if args.model_family == ModelFamily::Lighton && args.reasoning != ReasoningMode::Off {
            bail!("--reasoning is only supported with --family qianfan");
        }

        let selection = match args.model_family {
            ModelFamily::Lighton => ModelSelection::Lighton {
                spec: args.lighton_variant.spec(),
            },
            ModelFamily::Qianfan => ModelSelection::Qianfan {
                spec: args.qianfan_model.spec(),
                reasoning: args.reasoning,
            },
        };
        let defaults = selection.defaults();
        let long_edge = args.long_edge.unwrap_or(defaults.long_edge);
        let prompt = args.prompt.unwrap_or_else(|| defaults.prompt.to_owned());
        let max_tokens = args.max_tokens.unwrap_or(defaults.max_tokens);
        let n_ctx = args.n_ctx.unwrap_or(defaults.n_ctx);
        let n_batch = args.n_batch.unwrap_or(defaults.n_batch);
        let temperature = args.temperature.unwrap_or(defaults.temperature);

        let resolved = Self {
            selection,
            model_override: args.model,
            mmproj_override: args.mmproj,
            pdf_pages: args.pdf_pages,
            margin: args.margin,
            long_edge,
            threshold: args.threshold,
            render_dpi: args.render_dpi,
            detect_dpi: args.detect_dpi,
            autocrop: args.autocrop,
            dump_crops_dir: args.dump_crops_dir,
            export_detected_images_dir: args.export_detected_images_dir,
            output: args.output,
            output_template: args.output_template,
            live_preview: args.live_preview,
            show_llama_logs: args.show_llama_logs,
            prompt,
            max_tokens,
            n_ctx,
            n_batch,
            n_threads: args.n_threads,
            temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            seed: args.seed,
            no_gpu: args.no_gpu,
            print_prompt: args.print_prompt,
            inputs: args.inputs,
        };
        resolved.validate()?;
        Ok(resolved)
    }

    fn validate(&self) -> Result<()> {
        if self.long_edge == 0 {
            bail!("--long-edge must be greater than 0");
        }
        if let Some(max_long_edge) = self.selection.max_long_edge()
            && self.long_edge > max_long_edge
        {
            bail!(
                "--long-edge must be at most {max_long_edge} for {}, got {}",
                self.selection.family().display_name(),
                self.long_edge
            );
        }
        if self.margin.saturating_mul(2) >= self.long_edge {
            bail!("--margin must be less than half of --long-edge");
        }
        if self.render_dpi == 0 {
            bail!("--render-dpi must be greater than 0");
        }
        if self.detect_dpi == 0 {
            bail!("--detect-dpi must be greater than 0");
        }
        if self.export_detected_images_dir.is_some() && !self.selection.supports_bbox_exports() {
            bail!("{}", self.selection.export_requirement_hint());
        }
        if self.selection.family() == ModelFamily::Qianfan && self.prompt.trim().is_empty() {
            bail!("Qianfan OCR requires a non-empty prompt; pass --prompt or use the default");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::app::model::{
        DEFAULT_LONG_EDGE_LIGHTON, DEFAULT_LONG_EDGE_QIANFAN, DEFAULT_PROMPT_QIANFAN,
    };

    #[test]
    fn args_default_to_lighton_family() {
        let args = Args::try_parse_from(["pageocr", "input.pdf"]).unwrap();
        assert_eq!(args.model_family, ModelFamily::Lighton);
    }

    #[test]
    fn args_default_to_default_lighton_variant() {
        let args = Args::try_parse_from(["pageocr", "input.pdf"]).unwrap();
        assert_eq!(args.lighton_variant, LightonVariant::Default);
    }

    #[test]
    fn args_accept_bbox_lighton_variants() {
        let bbox = Args::try_parse_from(["pageocr", "input.pdf", "--variant", "bbox"]).unwrap();
        assert_eq!(bbox.lighton_variant, LightonVariant::Bbox);

        let bbox_soup =
            Args::try_parse_from(["pageocr", "input.pdf", "--variant", "bbox-soup"]).unwrap();
        assert_eq!(bbox_soup.lighton_variant, LightonVariant::BboxSoup);
    }

    #[test]
    fn args_accept_qianfan_family_flags() {
        let args = Args::try_parse_from([
            "pageocr",
            "input.png",
            "--family",
            "qianfan",
            "--qianfan-model",
            "bf16",
            "--reasoning",
            "on",
        ])
        .unwrap();

        assert_eq!(args.model_family, ModelFamily::Qianfan);
        assert_eq!(args.qianfan_model, QianfanModel::Bf16);
        assert_eq!(args.reasoning, ReasoningMode::On);
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
        assert_eq!(
            args.output_template.as_deref(),
            Some(Path::new("template.j2"))
        );
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
    fn resolved_args_use_lighton_defaults() {
        let args = ResolvedArgs::from_args(Args::try_parse_from(["pageocr", "input.pdf"]).unwrap())
            .unwrap();

        assert_eq!(args.long_edge, DEFAULT_LONG_EDGE_LIGHTON);
        assert_eq!(args.prompt, "");
        assert_eq!(args.max_tokens, 2048);
        assert_eq!(args.n_ctx, 4096);
        assert_eq!(args.n_batch, 512);
        assert_eq!(args.temperature, 0.2);
    }

    #[test]
    fn resolved_args_use_qianfan_defaults() {
        let args = ResolvedArgs::from_args(
            Args::try_parse_from(["pageocr", "--family", "qianfan", "input.png"]).unwrap(),
        )
        .unwrap();

        assert_eq!(args.long_edge, DEFAULT_LONG_EDGE_QIANFAN);
        assert_eq!(args.prompt, DEFAULT_PROMPT_QIANFAN);
        assert_eq!(args.max_tokens, 4096);
        assert_eq!(args.n_ctx, 8192);
        assert_eq!(args.n_batch, 1024);
        assert_eq!(args.temperature, 0.0);
    }

    #[test]
    fn lighton_rejects_reasoning_flag() {
        let err = ResolvedArgs::from_args(
            Args::try_parse_from(["pageocr", "--reasoning", "on", "input.pdf"]).unwrap(),
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("--reasoning is only supported with --family qianfan"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn extract_images_requires_bbox_capable_family() {
        let err = ResolvedArgs::from_args(
            Args::try_parse_from(["pageocr", "--extract-images-dir", "images", "input.pdf"])
                .unwrap(),
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("--extract-images-dir requires"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn qianfan_allows_detected_image_exports() {
        let args = ResolvedArgs::from_args(
            Args::try_parse_from([
                "pageocr",
                "--family",
                "qianfan",
                "--extract-images-dir",
                "images",
                "input.png",
            ])
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            args.export_detected_images_dir.as_deref(),
            Some(Path::new("images"))
        );
    }
}
