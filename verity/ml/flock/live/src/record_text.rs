//! A session record's canonical text: Lean's `Json.compress` of the record, which the verifier re-renders and compares.
//!
//! [`canonical_json`] is what every writer of a `flock-live-session/v1` record (`session.json`) emits: keys in code-point
//! order and no whitespace; strings escaped as Lean's `Json.escape` does (`"`, `\`, `\n`, `\r` with a backslash, every
//! other character below U+0020 as `\u00xx`, the rest as itself); an integer as itself, and a float as
//! `JsonNumber.toString` prints the number Lean's parser reads from its shortest round-trip text (`-191.0` → `-191`,
//! `1e-5` → `0.00001`). Python's `verity_flock.record_text.session_text` is the same function; both sides' tests hold
//! it to `record_text_vectors.json`, whose expected texts are Lean's.

use serde_json::Value;

/// `v` as Lean's `Json.compress` prints it.
pub fn canonical_json(v: &Value) -> String {
    let mut s = String::new();
    write_value(v, &mut s);
    s
}

fn write_value(v: &Value, s: &mut String) {
    match v {
        Value::Null => s.push_str("null"),
        Value::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) if n.is_f64() => s.push_str(&lean_number(&n.to_string())),
        Value::Number(n) => s.push_str(&n.to_string()),
        Value::String(x) => write_string(x, s),
        Value::Array(a) => {
            s.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                write_value(x, s);
            }
            s.push(']');
        }
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            s.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                write_string(k, s);
                s.push(':');
                write_value(&o[k], s);
            }
            s.push('}');
        }
    }
}

fn write_string(x: &str, s: &mut String) {
    s.push('"');
    for c in x.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            c if (c as u32) < 0x20 => s.push_str(&format!("\\u{:04x}", c as u32)),
            c => s.push(c),
        }
    }
    s.push('"');
}

/// `JsonNumber.toString` of the number Lean's `Json.parse` reads from `t` (a JSON number), on decimal digit strings:
/// the mantissa can be far past `u64` (`1e300`).
fn lean_number(t: &str) -> String {
    let neg = t.starts_with('-');
    let t = t.trim_start_matches('-').to_ascii_lowercase();
    let (body, exp) = match t.split_once('e') {
        Some((b, x)) => (b.to_string(), Some(x.parse::<i64>().expect("a JSON number's exponent"))),
        None => (t.clone(), None),
    };
    let (whole, frac) = body.split_once('.').unwrap_or((body.as_str(), ""));
    let mut digits = format!("{whole}{frac}").trim_start_matches('0').to_string();
    let mut e = frac.len() as i64;
    if let Some(k) = exp {
        if k >= 0 {
            if !digits.is_empty() {
                digits.push_str(&"0".repeat((k - e).max(0) as usize));
            }
            e = (e - k).max(0);
        } else {
            e -= k;
        }
    }
    let zero = digits.is_empty();
    if zero {
        digits = "0".to_string();
    }
    let sign = if neg && !zero { "-" } else { "" };
    if e == 0 {
        return format!("{sign}{digits}");
    }
    let n = digits.len() as i64;
    let x = (9 + n - e).min(0);
    let p = (e + x) as usize;
    let padded = format!("{}{digits}", "0".repeat(p.saturating_sub(digits.len())));
    let (left, rest) = padded.split_at(padded.len() - p);
    let left = if left.is_empty() { "0" } else { left };
    if rest.bytes().all(|b| b == b'0') && x == 0 {
        return format!("{sign}{left}");
    }
    let right = rest.trim_end_matches('0');
    let exp = if x == 0 { String::new() } else { format!("e{x}") };
    format!("{sign}{left}.{right}{exp}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cases() -> Vec<Value> {
        let doc: Value = serde_json::from_str(include_str!("record_text_vectors.json")).unwrap();
        let cases = doc["cases"].as_array().unwrap().clone();
        assert!(cases.iter().any(|c| c["name"] == "floats"));
        cases
    }

    /// The numbers of the `floats` case's `{"f": [...]}` text, one per element. serde_json's default float parser is
    /// not correctly rounded (it reads `-900821.3732204572` one ulp off) and refuses Lean's `0.0000000005e-314`, so
    /// these tests read them with `str::parse`, which is; a writer holds `f64`s and never parses them.
    fn numbers(text: &str) -> Vec<&str> {
        let inner = text.trim().strip_prefix("{\"f\":").unwrap().trim();
        let inner = inner.strip_prefix('[').unwrap().strip_suffix("]}").unwrap();
        inner.split(',').map(str::trim).collect()
    }

    #[test]
    fn matches_lean_on_the_shared_vectors() {
        for c in cases() {
            let (input, canonical) = (c["input"].as_str().unwrap(), c["canonical"].as_str().unwrap());
            let v: Value = if c["name"] == "floats" {
                let xs = numbers(input).into_iter().map(|t| {
                    Value::Number(serde_json::Number::from_f64(t.parse::<f64>().unwrap()).unwrap())
                });
                serde_json::json!({ "f": xs.collect::<Vec<_>>() })
            } else {
                serde_json::from_str(input).unwrap()
            };
            assert_eq!(canonical_json(&v), canonical, "case {}", c["name"]);
        }
    }

    #[test]
    fn reads_back_to_itself() {
        for c in cases() {
            let t = c["canonical"].as_str().unwrap();
            if c["name"] == "floats" {
                for x in numbers(t) {
                    assert_eq!(lean_number(x), x);
                }
            } else {
                assert_eq!(canonical_json(&serde_json::from_str(t).unwrap()), t, "case {}", c["name"]);
            }
        }
    }
}
