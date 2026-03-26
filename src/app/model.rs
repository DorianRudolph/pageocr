use std::{
    env, fs,
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Cache, Repo};
use llama_cpp_4::model::{LlamaChatMessage, LlamaModel};

pub const DEFAULT_PROMPT_QIANFAN: &str = r#"You are an AI assistant specialized in converting document images (one or multiple pages extracted from a PDF) into Markdown with high fidelity.

Your task is to accurately convert all visible content from the images into Markdown, strictly following the rules below. Do not add explanations, comments, or inferred content.

1. Text Recognition:
- Accurately convert all visible text.
- No guessing, inference, paraphrasing, or correction.
- Preserve the original document structure, including headings, paragraphs, lists, captions, and footnotes.
- Completely REMOVE all header and footer text. Do not output page numbers, running titles, or repeated marginal content.

2. Reading Order:
- Follow a top-to-bottom, left-to-right reading order.
- For multi-column layouts, fully read the left column before the right column.
- Do not reorder content for semantic or logical clarity.

3. Mathematical Formulas:
- Convert all mathematical expressions to LaTeX.
- Inline formulas must use $...$.
- Display (block) formulas must use:
  $$
  ...
  $$
- Preserve symbols, spacing, and structure exactly.
- Do not invent, simplify, normalize, or correct formulas.

4. Tables:
- Convert all tables to HTML format.
- Wrap each table with <table> and </table>.
- Preserve row and column order, merged cells (rowspan, colspan), and empty cells.
- Do not restructure or reinterpret tables.

5. Images:
- Do NOT describe image content.
- Preserve images using the exact format:
  ![label](<box>[[x1, y1, x2, y2]]</box>)
- Allowed labels: image, chart, seal.
- Completely REMOVE all header_image and footer_image elements.
- Do not introduce new labels.
- Do not remove or merge remaining image elements.

6. Unreadable or Missing Content:
- If text, symbols, or table cells are unreadable, preserve their position and leave the content empty.
- Do not guess or fill in missing information.

7. Output Requirements:
- Output Markdown only.
- Preserve original layout, spacing, and structure as closely as possible.
- Ensure clear separation between elements using line breaks.
- Do not include any explanations, metadata, or comments."#;
pub const DEFAULT_LONG_EDGE_LIGHTON: u32 = 1540;
pub const DEFAULT_LONG_EDGE_QIANFAN: u32 = 2048;
pub const MAX_LONG_EDGE_LIGHTON: u32 = DEFAULT_LONG_EDGE_LIGHTON;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModelFamily {
    Lighton,
    Qianfan,
}

impl ModelFamily {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Lighton => "LightOn",
            Self::Qianfan => "Qianfan",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LightonVariant {
    Default,
    Bbox,
    BboxSoup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum QianfanModel {
    Bf16,
    Q8,
}

impl QianfanModel {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q8 => "q8",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ReasoningMode {
    Off,
    On,
}

#[derive(Debug, Clone, Copy)]
pub struct InferenceDefaults {
    pub long_edge: u32,
    pub prompt: &'static str,
    pub max_tokens: usize,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub temperature: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum ModelSelection {
    Lighton {
        spec: LightonModelSpec,
    },
    Qianfan {
        spec: QianfanModelSpec,
        reasoning: ReasoningMode,
    },
}

impl ModelSelection {
    pub fn family(self) -> ModelFamily {
        match self {
            Self::Lighton { .. } => ModelFamily::Lighton,
            Self::Qianfan { .. } => ModelFamily::Qianfan,
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Lighton { spec } => spec.variant.display_name(),
            Self::Qianfan { spec, .. } => spec.variant.display_name(),
        }
    }

    pub fn supports_bbox_exports(self) -> bool {
        match self {
            Self::Lighton { spec } => spec.supports_bbox_exports,
            Self::Qianfan { spec, .. } => spec.supports_bbox_exports,
        }
    }

    pub fn defaults(self) -> InferenceDefaults {
        match self {
            Self::Lighton { spec } => spec.defaults(),
            Self::Qianfan { spec, .. } => spec.defaults(),
        }
    }

    pub fn max_long_edge(self) -> Option<u32> {
        match self {
            Self::Lighton { .. } => Some(MAX_LONG_EDGE_LIGHTON),
            Self::Qianfan { .. } => None,
        }
    }

    pub fn resolve_paths(
        self,
        model_override: Option<&Path>,
        mmproj_override: Option<&Path>,
    ) -> Result<(PathBuf, PathBuf)> {
        let cache_spec = self.cache_spec();
        let cache = model_cache();
        let default_paths = resolve_default_model_paths(
            &cache,
            cache_spec,
            model_override.is_none(),
            mmproj_override.is_none(),
            prompt_for_model_download,
            |spec, missing| download_model_files(&cache, spec, missing),
        )?;

        let model_path = model_override
            .map(normalize_path)
            .or(default_paths.model)
            .expect("default model path must be resolved");
        let mmproj_path = mmproj_override
            .map(normalize_path)
            .or(default_paths.mmproj)
            .expect("default mmproj path must be resolved");

        Ok((model_path, mmproj_path))
    }

    pub fn format_prompt(
        self,
        model: &LlamaModel,
        media_marker: &str,
        prompt: &str,
    ) -> Result<String> {
        match self {
            Self::Lighton { .. } => {
                let user_content = build_lighton_user_content(media_marker, prompt);
                let chat = vec![LlamaChatMessage::new("user".to_owned(), user_content)?];
                model
                    .apply_chat_template(None, chat, true)
                    .context("failed to apply the GGUF chat template")
            }
            Self::Qianfan { reasoning, .. } => {
                Ok(build_qianfan_prompt(media_marker, prompt, reasoning))
            }
        }
    }

    pub fn export_requirement_hint(self) -> &'static str {
        match self {
            Self::Lighton { .. } => "--images-dir requires --qianfan or --bbox/--bbox-soup",
            Self::Qianfan { .. } => "--images-dir requires --qianfan or --bbox/--bbox-soup",
        }
    }

    fn cache_spec(self) -> CachedModelSpec {
        match self {
            Self::Lighton { spec } => CachedModelSpec {
                family_name: "LightOn",
                variant_name: spec.variant.display_name(),
                repo_id: spec.repo_id,
                model_filename: spec.model_filename,
                mmproj_filename: spec.mmproj_filename,
            },
            Self::Qianfan { spec, .. } => CachedModelSpec {
                family_name: "Qianfan",
                variant_name: spec.variant.display_name(),
                repo_id: spec.repo_id,
                model_filename: spec.model_filename,
                mmproj_filename: spec.mmproj_filename,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LightonModelSpec {
    variant: LightonVariant,
    repo_id: &'static str,
    model_filename: &'static str,
    mmproj_filename: &'static str,
    supports_bbox_exports: bool,
}

impl LightonVariant {
    pub(super) fn spec(self) -> LightonModelSpec {
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

    pub fn display_name(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Bbox => "bbox",
            Self::BboxSoup => "bbox-soup",
        }
    }
}

impl LightonModelSpec {
    fn defaults(self) -> InferenceDefaults {
        InferenceDefaults {
            long_edge: DEFAULT_LONG_EDGE_LIGHTON,
            prompt: "",
            max_tokens: 2048,
            n_ctx: 4096,
            n_batch: 512,
            temperature: 0.2,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct QianfanModelSpec {
    variant: QianfanModel,
    repo_id: &'static str,
    model_filename: &'static str,
    mmproj_filename: &'static str,
    supports_bbox_exports: bool,
}

impl QianfanModel {
    pub(super) fn spec(self) -> QianfanModelSpec {
        match self {
            Self::Bf16 => QianfanModelSpec {
                variant: self,
                repo_id: "Reza2kn/Qianfan-OCR-GGUF",
                model_filename: "Qianfan-OCR-bf16.gguf",
                mmproj_filename: "Qianfan-OCR-mmproj-f16.gguf",
                supports_bbox_exports: true,
            },
            Self::Q8 => QianfanModelSpec {
                variant: self,
                repo_id: "Reza2kn/Qianfan-OCR-GGUF",
                model_filename: "Qianfan-OCR-q8_0.gguf",
                mmproj_filename: "Qianfan-OCR-mmproj-f16.gguf",
                supports_bbox_exports: true,
            },
        }
    }
}

impl QianfanModelSpec {
    fn defaults(self) -> InferenceDefaults {
        InferenceDefaults {
            long_edge: DEFAULT_LONG_EDGE_QIANFAN,
            prompt: DEFAULT_PROMPT_QIANFAN,
            max_tokens: 4096,
            n_ctx: 8192,
            n_batch: 1024,
            temperature: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CachedModelSpec {
    family_name: &'static str,
    variant_name: &'static str,
    repo_id: &'static str,
    model_filename: &'static str,
    mmproj_filename: &'static str,
}

#[derive(Debug, Default)]
struct DefaultModelPaths {
    model: Option<PathBuf>,
    mmproj: Option<PathBuf>,
}

fn build_lighton_user_content(media_marker: &str, prompt: &str) -> String {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        media_marker.to_owned()
    } else {
        format!("{media_marker}\n{prompt}")
    }
}

fn build_qianfan_prompt(media_marker: &str, prompt: &str, reasoning: ReasoningMode) -> String {
    let prompt = prompt.trim();
    let mut formatted = String::from("<|im_start|>user\n");
    formatted.push_str(media_marker);
    formatted.push_str(prompt);
    if reasoning == ReasoningMode::On {
        formatted.push_str("\n<think>");
    }
    formatted.push_str("<|im_end|>\n<|im_start|>assistant\n");
    formatted
}

fn resolve_default_model_paths<FPrompt, FDownload>(
    cache: &Cache,
    spec: CachedModelSpec,
    need_model: bool,
    need_mmproj: bool,
    mut prompt_for_download: FPrompt,
    mut download_missing: FDownload,
) -> Result<DefaultModelPaths>
where
    FPrompt: FnMut(CachedModelSpec, &[&'static str]) -> Result<bool>,
    FDownload: FnMut(CachedModelSpec, &[&'static str]) -> Result<Vec<(&'static str, PathBuf)>>,
{
    let mut paths = DefaultModelPaths::default();
    let mut missing = Vec::new();

    if need_model {
        if let Some(path) = cached_model_path(cache, spec, spec.model_filename) {
            paths.model = Some(normalize_path(&path));
        } else {
            missing.push(spec.model_filename);
        }
    }

    if need_mmproj {
        if let Some(path) = cached_model_path(cache, spec, spec.mmproj_filename) {
            paths.mmproj = Some(normalize_path(&path));
        } else {
            missing.push(spec.mmproj_filename);
        }
    }

    if missing.is_empty() {
        return Ok(paths);
    }

    if !prompt_for_download(spec, &missing)? {
        bail!("{}", declined_model_download_message(spec, &missing));
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

fn cached_model_path(
    cache: &Cache,
    spec: CachedModelSpec,
    filename: &'static str,
) -> Option<PathBuf> {
    cache
        .repo(model_repo(spec))
        .get(filename)
        .filter(|path| path.exists())
}

fn model_cache() -> Cache {
    env_path("HF_HUB_CACHE")
        .or_else(|| env_path("HUGGINGFACE_HUB_CACHE"))
        .map(Cache::new)
        .unwrap_or_else(Cache::from_env)
}

fn model_repo(spec: CachedModelSpec) -> Repo {
    Repo::model(spec.repo_id.to_owned())
}

fn env_path(key: &str) -> Option<PathBuf> {
    env::var_os(key)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn prompt_for_model_download(spec: CachedModelSpec, missing: &[&'static str]) -> Result<bool> {
    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        bail!("{}", noninteractive_model_download_message(spec, missing));
    }

    eprintln!(
        "{} {} model files are not cached:",
        spec.family_name, spec.variant_name
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

fn download_model_files(
    cache: &Cache,
    spec: CachedModelSpec,
    missing: &[&'static str],
) -> Result<Vec<(&'static str, PathBuf)>> {
    let api = ApiBuilder::from_cache(cache.clone())
        .with_progress(true)
        .build()
        .context("failed to initialize Hugging Face API client")?;
    let repo = api.repo(model_repo(spec));

    missing
        .iter()
        .map(|filename| {
            repo.download(filename)
                .with_context(|| format!("failed to download {filename}"))
                .map(|path| (*filename, path))
        })
        .collect()
}

fn noninteractive_model_download_message(
    spec: CachedModelSpec,
    missing: &[&'static str],
) -> String {
    format!(
        "{} {} model file(s) are missing from the Hugging Face cache: {}. Rerun in an interactive terminal to approve the download, or prefetch them with `hf download {} {}`.",
        spec.family_name,
        spec.variant_name,
        missing.join(", "),
        spec.repo_id,
        missing.join(" ")
    )
}

fn declined_model_download_message(spec: CachedModelSpec, missing: &[&'static str]) -> String {
    format!(
        "{} {} model download declined; required file(s) are missing from the Hugging Face cache: {}.",
        spec.family_name,
        spec.variant_name,
        missing.join(", ")
    )
}

fn normalize_path(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qianfan_default_prompt_omits_page_separator_step() {
        assert!(!DEFAULT_PROMPT_QIANFAN.contains("--- Page N ---"));
        assert!(!DEFAULT_PROMPT_QIANFAN.contains("If there are multiple pages"));
    }

    #[test]
    fn qianfan_prompt_without_reasoning_matches_chatml_shape() {
        let prompt =
            build_qianfan_prompt("<__media__>", DEFAULT_PROMPT_QIANFAN, ReasoningMode::Off);

        assert_eq!(
            prompt,
            format!(
                "<|im_start|>user\n<__media__>{DEFAULT_PROMPT_QIANFAN}<|im_end|>\n<|im_start|>assistant\n"
            )
        );
    }

    #[test]
    fn qianfan_prompt_with_reasoning_appends_think_to_last_user_turn() {
        let prompt = build_qianfan_prompt("<__media__>", DEFAULT_PROMPT_QIANFAN, ReasoningMode::On);

        assert_eq!(
            prompt,
            format!(
                "<|im_start|>user\n<__media__>{DEFAULT_PROMPT_QIANFAN}\n<think><|im_end|>\n<|im_start|>assistant\n"
            )
        );
    }
}
