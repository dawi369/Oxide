# Use the latest Rust slim image
FROM rust:slim-bookworm AS builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy project files
COPY . .

# Ensure the Data folder is available in the builder stage
# If Data is part of the project files, this step is redundant
# Otherwise, ensure Data is copied or created here
# COPY /path/to/local/Data /app/Data

# Update dependencies
RUN cargo update

# Build project
RUN cargo build --release

# Use minimal runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the built binary
COPY --from=builder /app/target/release/oxide /app/

# Copy the Data folder for runtime
COPY --from=builder /app/Data /app/Data

# Expose the application's port
EXPOSE 3000

# Run the application
CMD ["./oxide"]
