//! A minimal JSON reader for the proof headers (`proto/pcs.py` `Proof.to_bytes`, later proof.py's manifest
//! fields): numbers, strings, booleans, null, arrays, objects.  No dependency.

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
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
                t.parse::<f64>().map(Json::Num).map_err(|_| format!("json: bad number '{t}'"))
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
        assert_eq!(parse(" [true, null, \"a\\n\", -1.5e2] ").unwrap(), Json::Arr(vec![Json::Bool(true), Json::Null, Json::Str("a\n".into()), Json::Num(-150.0)]));
    }
}
