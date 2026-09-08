# Configuration boundaries

Commit experiment configuration only when every value is safe to disclose. Good
tracked values include workload shape, batch size, dtype, model identifier,
sampling settings, and benchmark repetition count.

Keep host-specific or sensitive values in ignored `config/local/` files or in the
environment. This includes API endpoints, SSH targets, provider credentials,
cloud account identifiers, and paths that identify a local user or machine.

Every executable should document the environment variable names it consumes, but
not their values. Prefer an explicit preflight validation step that lists missing
variable names and exits before starting work.
