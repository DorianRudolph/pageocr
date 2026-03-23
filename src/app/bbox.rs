use std::{
    env,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use anyhow::{Context, Result, bail};
use image::{RgbImage, imageops};
use pathdiff::diff_paths;
use regex::Regex;

const BBOX_PADDING: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BboxDetection {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

#[derive(Debug)]
struct ParsedBboxMatch {
    start: usize,
    end: usize,
    alt_text: String,
    detection: BboxDetection,
}

pub fn postprocess_markdown(
    markdown: String,
    source_image: &RgbImage,
    page_number: usize,
    supports_bbox_exports: bool,
    export_dir: Option<&Path>,
    output_path: Option<&Path>,
) -> Result<String> {
    if !supports_bbox_exports {
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
        output_path,
    )
}

pub fn export_bbox_detections(
    markdown: &str,
    source_image: &RgbImage,
    page_number: usize,
    export_dir: &Path,
    output_path: Option<&Path>,
) -> Result<String> {
    let detections = collect_bbox_matches(markdown);
    let mut rewritten = String::with_capacity(markdown.len());
    let mut last_end = 0usize;

    for (detection_index, matched) in detections.iter().enumerate() {
        rewritten.push_str(&markdown[last_end..matched.start]);

        let filename = format!("page_{page_number:04}_image_{:04}.png", detection_index + 1);
        let export_path = export_dir.join(&filename);

        match crop_bbox_detection(source_image, matched.detection) {
            Ok(crop) => {
                crop.save(&export_path).with_context(|| {
                    format!(
                        "failed to save detected image crop to {}",
                        export_path.display()
                    )
                })?;
                let link_path = markdown_export_path(&export_path, output_path)?;
                rewritten.push_str(&format!("![{}]({link_path})", matched.alt_text));
            }
            Err(err) => {
                log::debug!(
                    "skipping bbox detection on page {} due to invalid crop: {err:#}",
                    page_number
                );
                rewritten.push_str(&markdown[matched.start..matched.end]);
            }
        }

        last_end = matched.end;
    }

    rewritten.push_str(&markdown[last_end..]);
    Ok(rewritten)
}

fn collect_bbox_matches(markdown: &str) -> Vec<ParsedBboxMatch> {
    let mut matches = Vec::new();
    for captures in lighton_bbox_detection_pattern().captures_iter(markdown) {
        let matched = captures.get(0).expect("lighton bbox match must exist");
        matches.push(ParsedBboxMatch {
            start: matched.start(),
            end: matched.end(),
            alt_text: captures["alt"].to_owned(),
            detection: BboxDetection {
                left: captures["left"].parse().expect("bbox left must be numeric"),
                top: captures["top"].parse().expect("bbox top must be numeric"),
                right: captures["right"]
                    .parse()
                    .expect("bbox right must be numeric"),
                bottom: captures["bottom"]
                    .parse()
                    .expect("bbox bottom must be numeric"),
            },
        });
    }
    for captures in qianfan_bbox_detection_pattern().captures_iter(markdown) {
        let matched = captures.get(0).expect("qianfan bbox match must exist");
        matches.push(ParsedBboxMatch {
            start: matched.start(),
            end: matched.end(),
            alt_text: captures["alt"].to_owned(),
            detection: BboxDetection {
                left: captures["left"].parse().expect("bbox left must be numeric"),
                top: captures["top"].parse().expect("bbox top must be numeric"),
                right: captures["right"]
                    .parse()
                    .expect("bbox right must be numeric"),
                bottom: captures["bottom"]
                    .parse()
                    .expect("bbox bottom must be numeric"),
            },
        });
    }
    matches.sort_by_key(|matched| matched.start);
    matches
}

fn lighton_bbox_detection_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(
            r"!\[(?P<alt>[^\]]*)\]\((image_\d+\.png)\)\s*(?P<left>\d+)\s*,\s*(?P<top>\d+)\s*,\s*(?P<right>\d+)\s*,\s*(?P<bottom>\d+)",
        )
        .expect("LightOn bbox detection regex must compile")
    })
}

fn qianfan_bbox_detection_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(
            r"!\[(?P<alt>[^\]]*)\]\(<box>\s*\[\[\s*<COORD_(?P<left>\d+)>\s*,\s*<COORD_(?P<top>\d+)>\s*,\s*<COORD_(?P<right>\d+)>\s*,\s*<COORD_(?P<bottom>\d+)>\s*\]\]\s*</box>\)",
        )
        .expect("Qianfan bbox detection regex must compile")
    })
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

    let padded_left = left.saturating_sub(BBOX_PADDING);
    let padded_top = top.saturating_sub(BBOX_PADDING);
    let padded_right = right.saturating_add(BBOX_PADDING).min(width);
    let padded_bottom = bottom.saturating_add(BBOX_PADDING).min(height);

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

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(env::current_dir()
            .context("failed to read current working directory")?
            .join(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

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
            "![figure](image_1.png) 0,0,500,1000\n",
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

        assert!(rewritten.contains("![figure](../images/page_0003_image_0001.png)"));
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

    #[test]
    fn bbox_exports_accept_qianfan_markdown_format() -> Result<()> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!("ocr-cli-bbox-qianfan-test-{unique}"));
        let export_dir = root.join("images");
        fs::create_dir_all(&export_dir)?;

        let image = RgbImage::from_pixel(100, 100, image::Rgb([12, 34, 56]));
        let rewritten = export_bbox_detections(
            "![chart](<box>[[<COORD_100>, <COORD_200>, <COORD_800>, <COORD_900>]]</box>)",
            &image,
            1,
            &export_dir,
            None,
        )?;

        assert!(rewritten.starts_with("![chart]("));
        assert!(rewritten.ends_with("page_0001_image_0001.png)"));
        assert!(!rewritten.contains("<box>"));
        assert!(export_dir.join("page_0001_image_0001.png").exists());

        fs::remove_dir_all(root)?;
        Ok(())
    }
}
