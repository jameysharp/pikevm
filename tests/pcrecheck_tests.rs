use pikevm::compile;

static CASES: &str = include_str!("pcrecheck_tests");

#[test]
fn all_tests() {
    for (idx, line) in CASES.lines().enumerate() {
        let mut line = line.split_ascii_whitespace();
        let pattern = unquote(line.next().unwrap());
        let program = compile(pattern).unwrap();
        let input = unquote(line.next().unwrap());
        let results = program.exec(input.as_bytes());

        let mut matches = Vec::new();
        for (match_idx, found) in results {
            assert_eq!(match_idx, 0);
            matches = found;
        }

        let expected: Vec<usize> = line.map(|i| i.parse().unwrap()).collect();
        assert_eq!(
            matches,
            expected,
            "wrong captures for /{pattern}/ on '{input}' (line {})",
            idx + 1
        );
    }
}

fn unquote(s: &str) -> &str {
    &s[1..s.len() - 1]
}
