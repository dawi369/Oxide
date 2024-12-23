#![allow(warnings)]

// Models
mod gpt2;

// Frontend
use std::fs;

// Standard library imports
use std::convert::Infallible;
use std::net::SocketAddr;

// Hyper core imports
use hyper::{Method, Request, Response, StatusCode};
use hyper::service::service_fn;
use hyper::server::conn::http1;

// Hyper body-related imports
use hyper::body::{Body, Bytes, Frame};
use http_body_util::{combinators::BoxBody, BodyExt, Empty, Full};

// Runtime and IO imports
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

// Sync imports
use std::sync::Arc;
use tokio::sync::Mutex;

async fn hello(_: Request<hyper::body::Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    Ok(Response::new(Full::new(Bytes::from("Welcome to a lobotimized version of GPT-2\n"))))
}

// ROUTER FUNCTION ----------------------------------------------------------------------------------------------------------
async fn router(
    mut req: Request<hyper::body::Incoming>,
    model_handler: Arc<ort::Session>
) -> Result<Response<Full<Bytes>>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/") => {
            let html = fs::read_to_string("static/index.html").unwrap_or_else(|_| "File not found".to_string());
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/html")
                .body(Full::new(Bytes::from(html)))
                .unwrap())
        }
        (&Method::POST, "/inference") => {
            let body_bytes = req.body_mut().collect().await;
            match body_bytes {
                Ok(collected) => {
                    let input = String::from_utf8(collected.to_bytes().to_vec())
                        .unwrap_or_else(|_| "Invalid input".to_string());

                    // println!("Input: {:?}", input);
                    
                    // Handle the result of generate_text
                    match gpt2::generate_text(&model_handler, &input) {
                        Ok(output) => {
                            // println!("Output: {:?}", output);
                            Ok(Response::new(Full::new(Bytes::from(output))))
                        }
                        Err(_) => {
                            let mut response = Response::new(Full::new(Bytes::from("Internal Server Error")));
                            *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                            Ok(response)
                        }
                    }
                }
                Err(_) => {
                    let mut response = Response::new(Full::new(Bytes::from("Bad Request")));
                    *response.status_mut() = StatusCode::BAD_REQUEST;
                    Ok(response)
                }
            }
        }
        _ => {
            let mut response = Response::new(Full::new(Bytes::from("404 Not Found\n")));
            *response.status_mut() = StatusCode::NOT_FOUND;
            Ok(response)
        }
    }
}

// SERVER LOOP AND SHUTDOWN FUNCTION ----------------------------------------------------------------------------------------
async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));

    let model_handler = Arc::new(gpt2::initialize_session()?);

    let listener = TcpListener::bind(addr).await?;

    let mut http = http1::Builder::new();

    let graceful = hyper_util::server::graceful::GracefulShutdown::new();

    let mut signal = std::pin::pin!(shutdown_signal());

    // Server accept loop
    loop {
        tokio::select! {
            Ok((stream, _addr)) = listener.accept() => {
                let io = TokioIo::new(stream);
                let model_handler_clone = model_handler.clone();
                let conn = http.serve_connection(
                    io,
                    service_fn(move |req| router(req, model_handler_clone.clone()))
                );

                if let Err(e) = conn.await {
                    eprintln!("Error serving connection: {:?}", e);
                }
            },
            _ = &mut signal => {
                eprintln!("\nGraceful shutdown signal received");
                // Stop the accept loop
                break;
            }
        }
    }

    tokio::select! {
        _ = graceful.shutdown() => {
            eprintln!("All connections gracefully closed");
        },
        _ = tokio::time::sleep(std::time::Duration::from_secs(3)) => {
            eprintln!("Timed out waiting for all connections to close");
        }
    }

    Ok(())
}
