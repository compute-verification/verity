//! A minimal JSON reader for the proof headers (`proto/pcs.py` `Proof.to_bytes`, later proof.py's manifest
//! fields): numbers, strings, booleans, null, arrays, objects.  No dependency.

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// A plain integer literal (`0` or `-?[1-9][0-9]*`).
    Num(f64),
    /// Any other number token (`5.0`, `5e0`, `-0`, `007`): never accepted where an integer is expected.
    Real(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(x) if *x >= 0.0 && x.fract() == 0.0 && *x < 1e15 => Some(*x as usize),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }

    pub fn usize_list(&self) -> Option<Vec<usize>> {
        self.as_arr()?.iter().map(|x| x.as_usize()).collect()
    }

    /// Compact serialisation (integers without a fraction print as integers).
    pub fn dump(&self) -> String {
        let mut o = String::new();
        self.dump_into(&mut o);
        o
    }

    fn dump_into(&self, o: &mut String) {
        match self {
            Json::Null => o.push_str("null"),
            Json::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
            Json::Num(x) if x.fract() == 0.0 && x.abs() < 1e18 => o.push_str(&format!("{}", *x as i64)),
            Json::Num(x) => o.push_str(&format!("{x}")),
            Json::Real(x) if x.fract() == 0.0 && x.abs() < 1e18 => o.push_str(&format!("{x:.1}")),
            Json::Real(x) => o.push_str(&format!("{x:e}")),
            Json::Str(s) => {
                o.push('"');
                for ch in s.chars() {
                    match ch {
                        '"' => o.push_str("\\\""),
                        '\\' => o.push_str("\\\\"),
                        '\n' => o.push_str("\\n"),
                        c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                        c => o.push(c),
                    }
                }
                o.push('"');
            }
            Json::Arr(a) => {
                o.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        o.push(',');
                    }
                    x.dump_into(o);
                }
                o.push(']');
            }
            Json::Obj(kv) => {
                o.push('{');
                for (i, (k, v)) in kv.iter().enumerate() {
                    if i > 0 {
                        o.push(',');
                    }
                    Json::Str(k.clone()).dump_into(o);
                    o.push(':');
                    v.dump_into(o);
                }
                o.push('}');
            }
        }
    }
}

pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser { s: text.as_bytes(), pos: 0 };
    let v = p.value()?;
    p.ws();
    if p.pos != p.s.len() {
        return Err(format!("json: trailing characters at {}", p.pos));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.pos < self.s.len() && matches!(self.s[self.pos], b' ' | b'\n' | b'\r' | b'\t') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("json: expected '{}' at {}", c as char, self.pos))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        match self.peek() {
            None => Err("json: unexpected end".into()),
            Some(b'{') => {
                self.pos += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return Ok(Json::Obj(kv));
                }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    self.expect(b':')?;
                    let v = self.value()?;
                    kv.push((k, v));
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.pos += 1,
                        Some(b'}') => {
                            self.pos += 1;
                            return Ok(Json::Obj(kv));
                        }
                        _ => return Err(format!("json: expected ',' or '}}' at {}", self.pos)),
                    }
                }
            }
            Some(b'[') => {
                self.pos += 1;
                let mut a = Vec::new();
                self.ws();
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return Ok(Json::Arr(a));
                }
                loop {
                    a.push(self.value()?);
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.pos += 1,
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(Json::Arr(a));
                        }
                        _ => return Err(format!("json: expected ',' or ']' at {}", self.pos)),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') if self.s[self.pos..].starts_with(b"true") => {
                self.pos += 4;
                Ok(Json::Bool(true))
            }
            Some(b'f') if self.s[self.pos..].starts_with(b"false") => {
                self.pos += 5;
                Ok(Json::Bool(false))
            }
            Some(b'n') if self.s[self.pos..].starts_with(b"null") => {
                self.pos += 4;
                Ok(Json::Null)
            }
            Some(c) if c == b'-' || c.is_ascii_digit() => {
                let start = self.pos;
                while self.pos < self.s.len() && matches!(self.s[self.pos], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E') {
                    self.pos += 1;
                }
                let t = std::str::from_utf8(&self.s[start..self.pos]).map_err(|_| "json: number")?;
                let x = t.parse::<f64>().map_err(|_| format!("json: bad number '{t}'"))?;
                let d = t.strip_prefix('-').unwrap_or(t);
                let plain = !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()) && (d == "0" || !d.starts_with('0')) && t != "-0";
                Ok(if plain { Json::Num(x) } else { Json::Real(x) })
            }
            Some(c) => Err(format!("json: unexpected '{}' at {}", c as char, self.pos)),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                None => return Err("json: unterminated string".into()),
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let e = self.peek().ok_or("json: bad escape")?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'u' => {
                            let h = self.s.get(self.pos..self.pos + 4).ok_or("json: bad \\u")?;
                            let cp = u32::from_str_radix(std::str::from_utf8(h).map_err(|_| "json: \\u")?, 16).map_err(|_| "json: \\u")?;
                            self.pos += 4;
                            let ch = char::from_u32(cp).unwrap_or('\u{FFFD}');
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                        }
                        _ => return Err("json: bad escape".into()),
                    }
                }
                Some(c) => {
                    out.push(c);
                    self.pos += 1;
                }
            }
        }
        String::from_utf8(out).map_err(|_| "json: utf8".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_proto_header() {
        let h = r#"{"n":12,"kprime":[4,3,2],"queries":[16,16,16],"rate_log2":2,"splits":[[1024,1],[64,1],[16,1]],"coins":14,"hash":"blake3","beta":31,"ext_degree":6,"rows_shape":[[16,16],[16,48],[16,24]],"sib_len":[2048,640,256],"final_len":4}"#;
        let j = parse(h).unwrap();
        assert_eq!(j.get("n").unwrap().as_usize(), Some(12));
        assert_eq!(j.get("kprime").unwrap().usize_list(), Some(vec![4, 3, 2]));
        assert_eq!(j.get("hash").unwrap().as_str(), Some("blake3"));
        let splits: Vec<Vec<usize>> = j.get("splits").unwrap().as_arr().unwrap().iter().map(|x| x.usize_list().unwrap()).collect();
        assert_eq!(splits[0], vec![1024, 1]);
        assert!(parse("{\"a\": [1, 2,]}").is_err());
        assert!(parse("{\"a\": 1} x").is_err());
        assert_eq!(parse(" [true, null, \"a\\n\", -1.5e2] ").unwrap(), Json::Arr(vec![Json::Bool(true), Json::Null, Json::Str("a\n".into()), Json::Real(-150.0)]));
    }

    #[test]
    fn only_plain_integer_literals_are_integers() {
        for (t, int) in [("0", true), ("17", true), ("-3", true), ("5.0", false), ("5e0", false), ("-0", false), ("007", false), ("1E2", false)] {
            let j = parse(t).unwrap();
            assert_eq!(matches!(j, Json::Num(_)), int, "{t}");
        }
        assert_eq!(parse("5.0").unwrap().as_usize(), None);
    }
}
