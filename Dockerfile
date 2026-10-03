# Quokkaguard proxy image for Quome per-org deployment.
# Build context: repo root. Rules and chains are baked in; runtime config is
# mounted at /etc/qfire/config.toml (Secret Manager volume on Cloud Run).
FROM rust:1.88-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src/ ./src/
COPY benches/ ./benches/
COPY datasets/014-device-broker/devicebench/devices.json ./datasets/014-device-broker/devicebench/devices.json
RUN cargo build --release --bin qfire

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/qfire /usr/local/bin/qfire
COPY rules/ /app/rules/
COPY chains/ /app/chains/
WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/qfire"]
CMD ["serve", "--addr", "0.0.0.0:8080", "--config", "/etc/qfire/config.toml"]
