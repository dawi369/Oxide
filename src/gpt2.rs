use std::{
	io::{self, Write},
	path::Path
};

use ndarray::{Array1, ArrayViewD, Axis, array, concatenate, s};
use ort::{
    inputs,
    GraphOptimizationLevel,
    Session,
};

use rand::Rng;
use tokenizers::Tokenizer;

/// Max tokens to generate
const GEN_TOKENS: i32 = 90;
/// Top_K -> Sample from the k most likely next tokens at each step. Lower k focuses on higher probability tokens.
const TOP_K: usize = 5;

pub fn initialize_session() -> ort::Result<Session> {
    // Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
    ort::init()
        .with_name("GPT-2")
        .commit()?;

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Models").join("gpt2.onnx");

    // Load our model
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;

    Ok(session)
}

pub fn generate_text(session: &Session, prompt: &str) -> ort::Result<String> {
    let mut stdout = io::stdout();
    let mut rng = rand::thread_rng();

    // Load the tokenizer and encode the prompt into a sequence of tokens.
    let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("tokenizer.json")).unwrap();
    let tokens = tokenizer.encode(prompt, false).unwrap();
    let tokens = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

    let mut tokens = Array1::from_iter(tokens.iter().cloned());

    print!("{prompt}");
    stdout.flush().unwrap();

    let mut generated_text = String::new();

    for _ in 0..GEN_TOKENS {
        let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
        let outputs = session.run(inputs![array]?)?;
        let generated_tokens: ArrayViewD<f32> = outputs["output1"].try_extract_tensor()?;

        // Collect and sort logits
        let probabilities = &mut generated_tokens
            .slice(s![0, 0, -1, ..])
            .insert_axis(Axis(0))
            .to_owned()
            .iter()
            .cloned()
            .enumerate()
            .collect::<Vec<_>>();
        probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

        // Sample using top-k sampling
        let token = probabilities[rng.gen_range(0..=TOP_K)].0;
        tokens = concatenate![Axis(0), tokens, array![token.try_into().unwrap()]];

        let token_str = tokenizer.decode(&[token as _], true).unwrap();
        print!("{}", token_str);
        stdout.flush().unwrap();

        generated_text.push_str(&token_str);
    }

    println!();

    Ok(generated_text)
}

// fn main() -> ort::Result<()> {
//     // Initialize tracing to receive debug messages from `ort`
//     tracing_subscriber::fmt::init();

//     // Initialize the session once
//     let session = initialize_session()?;

//     // Generate text using the reusable session
//     generate_text(&session, PROMPT)?;

//     Ok(())
// }