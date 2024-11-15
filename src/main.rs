#![allow(warnings)]

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


async fn hello(_: Request<hyper::body::Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    Ok(Response::new(Full::new(Bytes::from("Hello, World!\n"))))
}


async fn handle_inference(req: Request<hyper::body::Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    // Get the path and query parameters
    let uri = req.uri();
    if let Some(query) = uri.query() {
        // Parse the number parameter
        if let Some(number_str) = query.split('=').nth(1) {
            if let Ok(number) = number_str.parse::<f64>() {
                let result = number.sqrt();
                return Ok(Response::new(Full::new(Bytes::from(format!("{}\n", result)))))
            }
        }
    }
    
    // Return 400 Bad Request if the input is invalid
    let mut response = Response::new(Full::new(Bytes::from("Invalid input. Use /inference?number=#\n")));
    *response.status_mut() = StatusCode::BAD_REQUEST;
    Ok(response)
}


async fn router(req: Request<hyper::body::Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/") => hello(req).await,
        (&Method::GET, "/inference") => handle_inference(req).await,
        _ => {
            let mut response = Response::new(Full::new(Bytes::from("404 Not Found\n")));
            *response.status_mut() = StatusCode::NOT_FOUND;
            Ok(response)
        }
    }
}



async fn shutdown_signal() {
    // Wait for the CTRL+C signal
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));

    // We create a TcpListener and bind it to 127.0.0.1:3000
    let listener = TcpListener::bind(addr).await?;
    // specify our HTTP settings (http1, http2, auto all work)
    let mut http = http1::Builder::new();
    // the graceful watcher
    let graceful = hyper_util::server::graceful::GracefulShutdown::new();
    // when this signal completes, start shutdown
    let mut signal = std::pin::pin!(shutdown_signal());

    // Our server accept loop
    loop {
        tokio::select! {
            Ok((stream, _addr)) = listener.accept() => {
                let io = TokioIo::new(stream);
                let conn = http.serve_connection(io, service_fn(router));
                // watch this connection
                let fut = graceful.watch(conn);
                tokio::spawn(async move {
                    if let Err(e) = fut.await {
                        eprintln!("Error serving connection: {:?}", e);
                    }
                });
            },

            _ = &mut signal => {
                eprintln!("\ngraceful shutdown signal received");
                // stop the accept loop
                break;
            }
        }
    }

    // Now start the shutdown and wait for them to complete
    // Optional: start a timeout to limit how long to wait.
    tokio::select! {
        _ = graceful.shutdown() => {
            eprintln!("all connections gracefully closed");
        },
        _ = tokio::time::sleep(std::time::Duration::from_secs(3)) => {
            eprintln!("timed out wait for all connections to close");
        }
    }

    Ok(())
}
