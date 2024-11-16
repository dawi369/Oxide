use ort::{Environment, SessionBuilder, Session, tensor::OrtOwnedTensor};
use hyper::{Response, StatusCode};
use http_body_util::Full;
use hyper::body::Bytes;
use std::sync::Arc;

// Model state structure
pub struct ModelHandler {
    session: Arc<Session>,
}

impl ModelHandler {
    pub async fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = load_model(model_path).await?;
        Ok(Self {
            session: Arc::new(session),
        })
    }

    pub async fn handle_inference(
        &self,
        input: String,
    ) -> Result<Response<Full<Bytes>>, Box<dyn std::error::Error>> {
        println!("Received input: {}", input);
        let response = Response::new(Full::new(Bytes::from("Inference handled")));
        Ok(response)
    }
    
}

async fn load_model(model_path: &str) -> Result<Session, Box<dyn std::error::Error>> {
    // Initialize the ONNX Runtime environment
    let environment = Arc::new(Environment::builder()
        .with_name("inference")
        .build()?);

    // Build and load the ONNX session
    let session = SessionBuilder::new(&environment)?
        .with_model_from_file(model_path)?;

    Ok(session)
}
