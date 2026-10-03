//! QFIRE binary entry point.

#[tokio::main]
async fn main() {
    // Logging is opt-in via RUST_LOG; default is quiet so CLI output stays clean.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    let code = qfire::cli::run().await;
    std::process::exit(code);
}
