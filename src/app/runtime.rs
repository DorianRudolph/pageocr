use std::num::NonZeroU32;

use anyhow::{Context, Result, bail};
use image::RgbImage;
use llama_cpp_4::{
    context::LlamaContext,
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{LlamaModel, Special, params::LlamaModelParams},
    mtmd::{MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunks, MtmdInputText},
    sampling::LlamaSampler,
};
use llama_cpp_sys_4 as llama_sys;

use super::cli::ResolvedArgs;

pub struct OcrRuntime {
    backend: LlamaBackend,
    model: LlamaModel,
    mtmd_ctx: MtmdContext,
    pub media_marker: String,
}

impl OcrRuntime {
    pub fn new(
        args: &ResolvedArgs,
        model_path: &std::path::Path,
        mmproj_path: &std::path::Path,
    ) -> Result<Self> {
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

    pub fn ocr_rgb_image(
        &self,
        image: &RgbImage,
        lctx: &mut LlamaContext<'_>,
        args: &ResolvedArgs,
    ) -> Result<String> {
        let bitmap = MtmdBitmap::from_rgb(image.width(), image.height(), image.as_raw())
            .context("failed to create bitmap from RGB image")?;
        self.ocr_bitmap(&bitmap, lctx, args)
    }

    pub fn new_context(&self, args: &ResolvedArgs) -> Result<LlamaContext<'_>> {
        let n_threads = resolved_n_threads(args.n_threads);
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(args.n_ctx))
            .with_n_batch(args.n_batch)
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);
        Ok(self.model.new_context(&self.backend, ctx_params)?)
    }

    fn ocr_bitmap(
        &self,
        bitmap: &MtmdBitmap,
        lctx: &mut LlamaContext<'_>,
        args: &ResolvedArgs,
    ) -> Result<String> {
        let formatted_prompt =
            args.selection
                .format_prompt(&self.model, &self.media_marker, &args.prompt)?;
        if args.print_prompt {
            log::debug!("formatted prompt:\n{formatted_prompt}");
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
}

pub fn resolved_n_threads(configured: Option<i32>) -> i32 {
    configured.filter(|&count| count > 0).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|count| count.get() as i32)
            .unwrap_or(1)
    })
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

fn build_sampler(args: &ResolvedArgs) -> LlamaSampler {
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

#[cfg(test)]
mod tests {
    use super::resolved_n_threads;

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
}
