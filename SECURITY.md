# Security Guidelines

## Security Improvements Implemented

### 1. Cryptographic Security
- **Fixed**: Replaced all `random` module usage with deterministic hash-based approaches for non-cryptographic mock data
- **Fixed**: Implemented secure secret key generation using `secrets.token_hex(32)` for 256-bit keys
- **Improvement**: All sensitive random operations now use cryptographically secure methods

### 2. Configuration Security  
- **Fixed**: Changed default host binding from `0.0.0.0` to `127.0.0.1` for better security posture
- **Fixed**: Removed hardcoded secret keys, now using environment variables with secure fallback
- **Best Practice**: Always set `SECRET_KEY` environment variable in production

### 3. Docker Security
- **Fixed**: Added non-root user `stockai` with proper UID/GID (1000:1000)
- **Fixed**: Proper file ownership and permissions for application directories
- **Fixed**: Health check dependencies properly installed
- **Best Practice**: Container runs as non-privileged user

### 4. Code Quality
- **Fixed**: Critical variable scope bugs that prevented application startup
- **Fixed**: Import resolution issues in module dependencies
- **Fixed**: Removed unused imports and variables
- **Improvement**: Deterministic behavior for testing and reproducibility

## Security Recommendations for Production

### Environment Variables
```bash
# Required
SECRET_KEY=your-256-bit-hex-key-here
DEBUG=false

# For network binding (deployment environments)
HOST=0.0.0.0  # Only set this in containerized deployments

# Database security
DB_PASSWORD=strong-database-password
REDIS_PASSWORD=strong-redis-password

# API keys (optional but recommended)
ALPHA_VANTAGE_API_KEY=your-api-key
```

### Docker Deployment
```bash
# Build with security context
docker build --no-cache -t stock-ai:secure .

# Run with security options
docker run --security-opt no-new-privileges \
           --read-only \
           --tmpfs /tmp \
           --user 1000:1000 \
           stock-ai:secure
```

### Security Monitoring
- Monitor for failed authentication attempts
- Log all API access patterns
- Implement rate limiting for public endpoints
- Regular security scans with `bandit` and `safety`

## Security Testing

Run security scans:
```bash
# Code security scan
bandit -r . -x tests/

# Dependency vulnerability scan  
pip-audit

# Docker image scan
docker scout cves stock-ai:latest
```

## Reporting Security Issues

If you discover a security vulnerability, please email security@stockai.com instead of opening a public issue.