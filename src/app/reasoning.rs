use std::sync::OnceLock;

use anyhow::Result;
use regex::Regex;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReasoningTraceEntry {
    pub page_number: usize,
    pub r#box: [u32; 4],
    pub label: String,
    pub brief: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedReasoning {
    pub markdown: String,
    pub entries: Vec<ReasoningTraceEntry>,
}

pub fn extract_reasoning_trace(markdown: String, page_number: usize) -> ExtractedReasoning {
    let mut cleaned = String::with_capacity(markdown.len());
    let mut entries = Vec::new();
    let mut cursor = 0usize;

    while let Some(relative_start) = markdown[cursor..].find("<think>") {
        let block_start = cursor + relative_start;
        let body_start = block_start + "<think>".len();
        cleaned.push_str(&markdown[cursor..block_start]);

        if let Some(relative_end) = markdown[body_start..].find("</think>") {
            let body_end = body_start + relative_end;
            entries.extend(parse_reasoning_entries(&markdown[body_start..body_end], page_number));
            cursor = body_end + "</think>".len();
        } else {
            entries.extend(parse_reasoning_entries(&markdown[body_start..], page_number));
            cursor = markdown.len();
            break;
        }
    }

    cleaned.push_str(&markdown[cursor..]);

    ExtractedReasoning {
        markdown: cleaned.trim_start_matches('\n').to_owned(),
        entries,
    }
}

pub fn render_reasoning_json(entries: &[ReasoningTraceEntry]) -> Result<String> {
    Ok(serde_json::to_string_pretty(entries)?)
}

fn parse_reasoning_entries(body: &str, page_number: usize) -> Vec<ReasoningTraceEntry> {
    let mut entries = Vec::new();
    for captures in reasoning_entry_pattern().captures_iter(body) {
        entries.push(ReasoningTraceEntry {
            page_number,
            r#box: [
                captures["left"].parse().expect("bbox left must be numeric"),
                captures["top"].parse().expect("bbox top must be numeric"),
                captures["right"]
                    .parse()
                    .expect("bbox right must be numeric"),
                captures["bottom"]
                    .parse()
                    .expect("bbox bottom must be numeric"),
            ],
            label: captures["label"].trim().to_owned(),
            brief: captures["brief"].trim().to_owned(),
        });
    }
    entries
}

fn reasoning_entry_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(
            r"(?s)<box>\s*\[\[\s*<COORD_(?P<left>\d+)>\s*,\s*<COORD_(?P<top>\d+)>\s*,\s*<COORD_(?P<right>\d+)>\s*,\s*<COORD_(?P<bottom>\d+)>\s*\]\]\s*</box>\s*<label>\s*(?P<label>.*?)\s*</label>\s*<brief>\s*(?P<brief>.*?)\s*</brief>",
        )
        .expect("reasoning entry regex must compile")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_reasoning_trace_removes_think_block_and_parses_entries() {
        let extracted = extract_reasoning_trace(
            "<think>\n<box>[[<COORD_001>, <COORD_002>, <COORD_003>, <COORD_004>]]</box>\n<label>text</label>\n<brief>Hello world.</brief>\n</think>\nBody".to_owned(),
            7,
        );

        assert_eq!(extracted.markdown, "Body");
        assert_eq!(
            extracted.entries,
            vec![ReasoningTraceEntry {
                page_number: 7,
                r#box: [1, 2, 3, 4],
                label: "text".to_owned(),
                brief: "Hello world.".to_owned(),
            }]
        );
    }

    #[test]
    fn extract_reasoning_trace_keeps_markdown_when_no_think_block_exists() {
        let extracted = extract_reasoning_trace("Plain markdown".to_owned(), 1);

        assert_eq!(extracted.markdown, "Plain markdown");
        assert!(extracted.entries.is_empty());
    }

    #[test]
    fn extract_reasoning_trace_removes_truncated_think_block_and_parses_entries() {
        let extracted = extract_reasoning_trace(
            "Body\n<think>\n<box>[[<COORD_010>, <COORD_020>, <COORD_030>, <COORD_040>]]</box>\n<label>image</label>\n<brief>Diagram.</brief>\n".to_owned(),
            3,
        );

        assert_eq!(extracted.markdown, "Body\n");
        assert_eq!(
            extracted.entries,
            vec![ReasoningTraceEntry {
                page_number: 3,
                r#box: [10, 20, 30, 40],
                label: "image".to_owned(),
                brief: "Diagram.".to_owned(),
            }]
        );
    }

    #[test]
    fn render_reasoning_json_uses_flat_entry_shape() {
        let rendered = render_reasoning_json(&[ReasoningTraceEntry {
            page_number: 2,
            r#box: [10, 20, 30, 40],
            label: "chart".to_owned(),
            brief: "Oscillation plot".to_owned(),
        }])
        .unwrap();

        assert!(rendered.contains("\"page_number\": 2"));
        assert!(rendered.contains("\"box\": ["));
        assert!(rendered.contains("\"label\": \"chart\""));
        assert!(rendered.contains("\"brief\": \"Oscillation plot\""));
    }
}
