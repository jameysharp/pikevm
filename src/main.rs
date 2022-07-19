use pikevm::compile_set;

fn main() -> regex_syntax::Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let (input, patterns) = args.split_last().unwrap();
    let input_bytes = input.as_bytes();
    for (idx, captures) in compile_set(patterns)?.exec(input_bytes) {
        println!(
            "MATCH: pattern '{}', input '{}'",
            patterns[idx as usize], input
        );
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
    }
    Ok(())
}
