use std::{
    io::{self, Write},
    path::Path,
};

use ndarray::{Array1, ArrayViewD, Axis, array, concatenate, s};
use ort::{
    inputs,
    GraphOptimizationLevel,
    Session,
};

use rand::Rng;
use tokenizers::Tokenizer;

const GEN_TOKENS: usize = 100; 
const TOP_K: usize = 5;

pub fn initialize_session() -> ort::Result<Session> {
    ort::init()
        .with_name("GPT-2")
        .commit()?;

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Data").join("gpt2.onnx");

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

    let tokenizer = Tokenizer::from_file(Path::new(env!("CARGO_MANIFEST_DIR")).join("Data").join("tokenizer.json")).unwrap();
    let encoded = tokenizer.encode(prompt, false).unwrap();
    let tokens = encoded.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>();

    let mut tokens = Array1::from_iter(tokens.iter().cloned());
    let mut generated_text = String::new();

    for _ in 0..GEN_TOKENS {
        let array = tokens.view().insert_axis(Axis(0)).insert_axis(Axis(1));
        let outputs = session.run(inputs![array]?)?;
        let generated_tokens: ArrayViewD<f32> = outputs["output1"].try_extract_tensor()?;

        let probabilities = &mut generated_tokens
            .slice(s![0, 0, -1, ..])
            .to_owned()
            .iter()
            .cloned()
            .enumerate()
            .collect::<Vec<_>>();
        probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Less));

        let token = probabilities.iter().take(TOP_K).max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0;
        tokens = concatenate![Axis(0), tokens, array![token as i64]];

        let token_str = tokenizer.decode(&[token as _], true).unwrap();

        generated_text.push_str(&token_str);

        if token_str.ends_with('.') || generated_text.split_whitespace().count() >= GEN_TOKENS {
            break;
        }
    }

    let truncated_text = generated_text.split_whitespace().take(GEN_TOKENS).collect::<Vec<_>>().join(" ");
    // println!("{}", truncated_text);

    Ok(truncated_text)
}
