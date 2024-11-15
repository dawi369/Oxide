use std::convert::Infallible;
use std::net::SocketAddr;

use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use hyper::body::{Frame, Body};
use hyper::{Method, StatusCode};
use http_body_util::{combinators::BoxBody, BodyExt, Empty, Full};


async fn hello(_: Request<hyper::body::Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    Ok(Response::new(Full::new(Bytes::from("Hello, World!"))))
}


async fn echo(
    req: Request<hyper::body::Incoming>,
) -> Result<Response<BoxBody<Bytes, hyper::Error>>, hyper::Error> {
    match (req.method(), req.uri().path()) {

        (&Method::GET, "/") => Ok(Response::new(full(
            "Try POSTing data to /echo\n",
        ))),

        // (&Method::POST, "/echo") => Ok(Response::new(req.into_body().boxed())),
        (&Method::POST, "/echo") => Ok(Response::new(req.into_body().boxed())),


        (&Method::POST, "/echo/uppercase") => {
            // Map this body's frame to a different type
            let frame_stream = req.into_body().map_frame(|frame| {
                let frame = if let Ok(data) = frame.into_data() {
                    let uppercase = data.iter()
                        .map(|byte| byte.to_ascii_uppercase())
                        .collect::<Bytes>();
                    println!("Processed {} bytes", uppercase.len()); // Add logging
                    uppercase
                } else {
                    Bytes::new()
                };
        
                Frame::data(frame)
            });

            Ok(Response::new(frame_stream.boxed()))
        },

        (&Method::POST, "/echo/reversed") => {
            // Protect our server from massive bodies.
            let upper = req.body().size_hint().upper().unwrap_or(u64::MAX);
            if upper > 1024 * 64 {
                let mut resp = Response::new(full("Body too big\n"));
                *resp.status_mut() = hyper::StatusCode::PAYLOAD_TOO_LARGE;
                return Ok(resp);
            }
        
            // Await the whole body to be collected into a single `Bytes`...
            let whole_body = req.collect().await?.to_bytes();
        
            // Iterate the whole body in reverse order and collect into a new Vec.
            let mut reversed_body = whole_body.iter()
                .rev()
                .cloned()
                .collect::<Vec<u8>>();
        
            // Add newline to the end
            reversed_body.push(b'\n');
        
            Ok(Response::new(full(reversed_body)))
        },

        // Return 404 Not Found for other routes.
        _ => {
            let mut not_found = Response::new(full("Invalid request path\n"));
            *not_found.status_mut() = StatusCode::NOT_FOUND;
            Ok(not_found)
        }
    }
}


// We create some utility functions to make Empty and Full bodies
// fit our broadened Response body type.
fn empty() -> BoxBody<Bytes, hyper::Error> {
    Empty::<Bytes>::new()
        .map_err(|never| match never {})
        .boxed()
}
fn full<T: Into<Bytes>>(chunk: T) -> BoxBody<Bytes, hyper::Error> {
    Full::new(chunk.into())
        .map_err(|never| match never {})
        .boxed()
}


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));

    // We create a TcpListener and bind it to 127.0.0.1:3000
    let listener = TcpListener::bind(addr).await?;

    // We start a loop to continuously accept incoming connections
    loop {
        let (stream, _) = listener.accept().await?;

        // Use an adapter to access something implementing `tokio::io` traits as if they implement
        // `hyper::rt` IO traits.
        let io = TokioIo::new(stream);

        // Spawn a tokio task to serve multiple connections concurrently
        tokio::task::spawn(async move {
            // Finally, we bind the incoming connection to our `hello` service
            if let Err(err) = http1::Builder::new()
                // `service_fn` converts our function in a `Service`
                .serve_connection(io, service_fn(echo))
                .await
            {
                eprintln!("Error serving connection: {:?}", err);
            }
        });
    }
}