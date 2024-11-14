# Stage 1: Build the Rust application
FROM rust:latest AS builder

# Add the musl target for building a statically linked binary
RUN rustup target add x86_64-unknown-linux-musl

# Install required dependencies for static linking
RUN apt update && apt install -y musl-tools musl-dev build-essential

# Set the working directory inside the container
WORKDIR /app

# Copy the source code into the container
COPY . .

# Set Rust flags to use the musl linker
ENV RUSTFLAGS='-C linker=musl-gcc'

# Build the application in release mode for the musl target to produce a static binary
RUN cargo build --release --target x86_64-unknown-linux-musl

# Stage 2: Run the application in a minimal image
FROM scratch

# Set environment variables (customize based on your server requirements)
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080

# Expose the port the server will listen on
EXPOSE 8080

# Copy the statically compiled binary from the builder stage
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/your_binary_name /app/server

# Set the default command to run the server
CMD ["/app/server"]
