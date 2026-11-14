//! The expression language of composite filters: numbers, macro names, `+ - * /`, parentheses
//! and a few functions. A parameter written as `"lerp(80, 400, telephone)"` follows its macro.
//!
//! An expression is compiled once (when the filter file is read) into postfix operations.
//! Evaluating it touches no heap, so a macro change can be applied on the audio thread; it runs
//! when a macro moves, never per sample.

use std::fmt::Write as _;

/// Deepest value stack an expression may need.
const MAX_STACK: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Func {
    Lerp,
    Clamp,
    Exp,
    Ln,
    Pow,
    Sqrt,
    Min,
    Max,
    Abs,
    Db,
}

const FUNCS: &[(&str, Func, usize)] = &[
    ("lerp", Func::Lerp, 3),
    ("clamp", Func::Clamp, 3),
    ("exp", Func::Exp, 1),
    ("ln", Func::Ln, 1),
    ("pow", Func::Pow, 2),
    ("sqrt", Func::Sqrt, 1),
    ("min", Func::Min, 2),
    ("max", Func::Max, 2),
    ("abs", Func::Abs, 1),
    ("db", Func::Db, 1),
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Num(f32),
    Var(usize),
    Add,
    Sub,
    Mul,
    Div,
    Neg,
    Call(Func),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    ops: Vec<Op>,
}

impl Expr {
    pub fn constant(v: f32) -> Self {
        Self { ops: vec![Op::Num(v)] }
    }

    /// Compile `src`. `vars` are the names it may refer to (the filter's macros), in the order
    /// [`eval`](Self::eval) receives their values.
    pub fn parse(src: &str, vars: &[&str]) -> Result<Self, String> {
        let mut p = Parser { src: src.as_bytes(), pos: 0, vars, ops: Vec::new() };
        p.expr()?;
        p.skip_ws();
        if p.pos < p.src.len() {
            return Err(p.unexpected());
        }
        let ops = p.ops;
        // Every operation's effect on the stack is known, so the deepest point is too.
        let (mut depth, mut deepest) = (0usize, 0usize);
        for op in &ops {
            match op {
                Op::Num(_) | Op::Var(_) => depth += 1,
                Op::Neg => {}
                Op::Add | Op::Sub | Op::Mul | Op::Div => depth -= 1,
                Op::Call(f) => depth -= FUNCS.iter().find(|x| x.1 == *f).map_or(0, |x| x.2 - 1),
            }
            deepest = deepest.max(depth);
        }
        if deepest > MAX_STACK {
            return Err("expression is nested too deeply".into());
        }
        Ok(Self { ops })
    }

    /// Whether the result depends on variable `i`.
    pub fn uses(&self, i: usize) -> bool {
        self.ops.contains(&Op::Var(i))
    }

    /// The value for these variable values. A variable beyond `vars` reads as 0.
    pub fn eval(&self, vars: &[f32]) -> f32 {
        let mut st = [0f32; MAX_STACK];
        let mut n = 0usize;
        for op in &self.ops {
            match *op {
                Op::Num(v) => {
                    st[n] = v;
                    n += 1;
                }
                Op::Var(i) => {
                    st[n] = vars.get(i).copied().unwrap_or(0.0);
                    n += 1;
                }
                Op::Neg => st[n - 1] = -st[n - 1],
                Op::Add | Op::Sub | Op::Mul | Op::Div => {
                    let (a, b) = (st[n - 2], st[n - 1]);
                    n -= 1;
                    st[n - 1] = match *op {
                        Op::Add => a + b,
                        Op::Sub => a - b,
                        Op::Mul => a * b,
                        _ => a / b,
                    };
                }
                Op::Call(f) => {
                    let argc = FUNCS.iter().find(|x| x.1 == f).map_or(1, |x| x.2);
                    let a = &st[n - argc..n];
                    let v = match f {
                        Func::Lerp => a[0] + (a[1] - a[0]) * a[2],
                        Func::Clamp => a[0].max(a[1].min(a[2])).min(a[1].max(a[2])),
                        Func::Exp => a[0].exp(),
                        Func::Ln => a[0].ln(),
                        Func::Pow => a[0].powf(a[1]),
                        Func::Sqrt => a[0].sqrt(),
                        Func::Min => a[0].min(a[1]),
                        Func::Max => a[0].max(a[1]),
                        Func::Abs => a[0].abs(),
                        Func::Db => 10f32.powf(a[0] / 20.0),
                    };
                    n -= argc;
                    st[n] = v;
                    n += 1;
                }
            }
        }
        st[0]
    }
}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    vars: &'a [&'a str],
    ops: Vec<Op>,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while self.src.get(self.pos).is_some_and(|c| c.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.src.get(self.pos).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn unexpected(&mut self) -> String {
        match self.peek() {
            Some(c) => format!("unexpected '{}' at column {}", c as char, self.pos + 1),
            None => "the expression ends too early".into(),
        }
    }

    fn expr(&mut self) -> Result<(), String> {
        self.term()?;
        loop {
            let op = match self.peek() {
                Some(b'+') => Op::Add,
                Some(b'-') => Op::Sub,
                _ => return Ok(()),
            };
            self.pos += 1;
            self.term()?;
            self.ops.push(op);
        }
    }

    fn term(&mut self) -> Result<(), String> {
        self.unary()?;
        loop {
            let op = match self.peek() {
                Some(b'*') => Op::Mul,
                Some(b'/') => Op::Div,
                _ => return Ok(()),
            };
            self.pos += 1;
            self.unary()?;
            self.ops.push(op);
        }
    }

    fn unary(&mut self) -> Result<(), String> {
        if self.eat(b'-') {
            self.unary()?;
            self.ops.push(Op::Neg);
            return Ok(());
        }
        self.primary()
    }

    fn primary(&mut self) -> Result<(), String> {
        let Some(c) = self.peek() else { return Err(self.unexpected()) };
        if c == b'(' {
            self.pos += 1;
            self.expr()?;
            return if self.eat(b')') { Ok(()) } else { Err(self.unexpected()) };
        }
        let start = self.pos;
        if c.is_ascii_digit() || c == b'.' {
            while self.src.get(self.pos).is_some_and(|c| c.is_ascii_digit() || *c == b'.') {
                self.pos += 1;
            }
            // An exponent, as in 1e-3.
            if self.src.get(self.pos).is_some_and(|c| *c == b'e' || *c == b'E') {
                let mut p = self.pos + 1;
                if self.src.get(p).is_some_and(|c| *c == b'+' || *c == b'-') {
                    p += 1;
                }
                if self.src.get(p).is_some_and(u8::is_ascii_digit) {
                    while self.src.get(p).is_some_and(u8::is_ascii_digit) {
                        p += 1;
                    }
                    self.pos = p;
                }
            }
            let text = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
            let v: f32 = text.parse().map_err(|_| format!("'{text}' is not a number"))?;
            self.ops.push(Op::Num(v));
            return Ok(());
        }
        if !(c.is_ascii_alphabetic() || c == b'_') {
            return Err(self.unexpected());
        }
        while self.src.get(self.pos).is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_') {
            self.pos += 1;
        }
        let name = std::str::from_utf8(&self.src[start..self.pos]).unwrap_or("");
        if self.eat(b'(') {
            let Some(&(_, func, argc)) = FUNCS.iter().find(|f| f.0 == name) else {
                let known: Vec<&str> = FUNCS.iter().map(|f| f.0).collect();
                return Err(format!("unknown function '{name}' (there are: {})", known.join(", ")));
            };
            let mut got = 0;
            if !self.eat(b')') {
                loop {
                    self.expr()?;
                    got += 1;
                    if self.eat(b')') {
                        break;
                    }
                    if !self.eat(b',') {
                        return Err(self.unexpected());
                    }
                }
            }
            if got != argc {
                return Err(format!("{name}() takes {argc} value{}, got {got}", if argc == 1 { "" } else { "s" }));
            }
            self.ops.push(Op::Call(func));
            return Ok(());
        }
        match self.vars.iter().position(|v| *v == name) {
            Some(i) => self.ops.push(Op::Var(i)),
            None => {
                let mut msg = format!("unknown name '{name}'");
                if self.vars.is_empty() {
                    msg.push_str(" (this filter has no macros)");
                } else {
                    let _ = write!(msg, " (macros: {})", self.vars.join(", "));
                }
                return Err(msg);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(src: &str, vars: &[(&str, f32)]) -> f32 {
        let names: Vec<&str> = vars.iter().map(|v| v.0).collect();
        let vals: Vec<f32> = vars.iter().map(|v| v.1).collect();
        Expr::parse(src, &names).unwrap_or_else(|e| panic!("{src}: {e}")).eval(&vals)
    }

    #[test]
    fn arithmetic_follows_the_usual_rules() {
        assert_eq!(eval("1 + 2 * 3", &[]), 7.0);
        assert_eq!(eval("(1 + 2) * 3", &[]), 9.0);
        assert_eq!(eval("10 - 4 - 3", &[]), 3.0);
        assert_eq!(eval("8 / 4 / 2", &[]), 1.0);
        assert_eq!(eval("-2 * -3", &[]), 6.0);
        assert_eq!(eval("2 - -1", &[]), 3.0);
        assert_eq!(eval("1.5e3 + .5", &[]), 1500.5);
        assert_eq!(eval("  4*strength ", &[("strength", 0.25)]), 1.0);
    }

    #[test]
    fn the_bad_mic_expressions_evaluate() {
        let m = [("strength", 0.5), ("telephone", 1.0), ("static", 0.0)];
        assert_eq!(eval("lerp(80, 400, telephone)", &m), 400.0);
        assert_eq!(eval("lerp(12000, 3400, telephone)", &m), 3400.0);
        assert_eq!(eval("6 * strength", &m), 3.0);
        assert_eq!(eval("lerp(16, 8, strength)", &m), 12.0);
        assert_eq!(eval("lerp(-80, -40, static)", &m), -80.0);
        assert_eq!(eval("4 * static", &m), 0.0);
    }

    #[test]
    fn functions() {
        assert_eq!(eval("clamp(5, 0, 1)", &[]), 1.0);
        assert_eq!(eval("clamp(-5, 0, 1)", &[]), 0.0);
        assert_eq!(eval("clamp(0.3, 1, 0)", &[]), 0.3); // bounds in either order
        assert!((eval("exp(1)", &[]) - std::f32::consts::E).abs() < 1e-6);
        assert!((eval("ln(exp(2))", &[]) - 2.0).abs() < 1e-6);
        assert_eq!(eval("pow(2, 10)", &[]), 1024.0);
        assert_eq!(eval("sqrt(16) + abs(-1)", &[]), 5.0);
        assert_eq!(eval("min(3, max(1, 2))", &[]), 2.0);
        assert!((eval("db(-6)", &[]) - 0.501).abs() < 1e-3);
        assert_eq!(eval("lerp(lerp(0, 10, a), 100, a)", &[("a", 0.5)]), 52.5);
    }

    #[test]
    fn mistakes_are_explained() {
        let e = |src: &str| Expr::parse(src, &["strength", "telephone"]).unwrap_err();
        assert_eq!(e("6 * strenght"), "unknown name 'strenght' (macros: strength, telephone)");
        assert_eq!(e("lerp(1, 2)"), "lerp() takes 3 values, got 2");
        assert_eq!(e("exp(1, 2)"), "exp() takes 1 value, got 2");
        assert!(e("lirp(1, 2, 3)").starts_with("unknown function 'lirp' (there are: lerp, clamp"));
        assert_eq!(e("1 +"), "the expression ends too early");
        assert_eq!(e("(1 + 2"), "the expression ends too early");
        assert_eq!(e("1 + 2)"), "unexpected ')' at column 6");
        assert_eq!(e("1 $ 2"), "unexpected '$' at column 3");
        assert_eq!(e("1.2.3"), "'1.2.3' is not a number");
        assert_eq!(Expr::parse("x", &[]).unwrap_err(), "unknown name 'x' (this filter has no macros)");
        let deep = format!("{}1{}", "(1+".repeat(40), ")".repeat(40));
        assert_eq!(Expr::parse(&deep, &[]).unwrap_err(), "expression is nested too deeply");
    }

    #[test]
    fn knows_what_it_depends_on() {
        let x = Expr::parse("lerp(1, 4, strength)", &["telephone", "strength"]).unwrap();
        assert!(x.uses(1) && !x.uses(0));
        assert!(!Expr::constant(3.0).uses(0));
        assert_eq!(Expr::constant(3.0).eval(&[]), 3.0);
    }
}
