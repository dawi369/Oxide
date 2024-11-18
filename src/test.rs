use ort::session::{Session, builder::GraphOptimizationLevel};

fn main() {
    let session = Session::builder()
        .with_model_from_file("path/to/your/model.onnx")
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Basic)
        .unwrap()
        .build()
        .unwrap();

    println!("Session created successfully!");
}