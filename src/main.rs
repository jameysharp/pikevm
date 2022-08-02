use pikevm::compile;

fn main() -> regex_syntax::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (input, patterns) = args.split_last().unwrap();
    let compiled = patterns
        .iter()
        .map(|pattern| compile(&pattern))
        .collect::<Result<Vec<_>, _>>()?;
    let input_bytes = input.as_bytes();

    for (pattern, program) in patterns.iter().zip(compiled) {
        let dfa = program.to_dfa();
        eprintln!("{}", dfa.to_dot());
        let captures = dfa.exec(input_bytes);
        if let Some(captures) = captures {
            println!("MATCH: pattern '{}', input '{}'", pattern, input);
            for (idx, group) in captures.chunks_exact(2).enumerate() {
                print!(" -- {}: ", idx);
                let start = group[0];
                let end = group[1];
                if start == usize::MAX {
                    println!("no match");
                } else {
                    println!(
                        "\"{}\" ({},{})",
                        String::from_utf8_lossy(&input_bytes[start..end]),
                        start,
                        end
                    );
                }
            }
        } else {
            println!("NO MATCH: pattern '{}', input '{}'", pattern, input);
        }
    }
    Ok(())
}
