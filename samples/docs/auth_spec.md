# Requirement Specification - Authentication

## Auth

### Login lockout
- Account is locked after 5 consecutive failed login attempts
- A locked user sees message "Your account is locked. Contact support."
- A successful login resets the failed attempt counter

### OTP for high-value transfers
- Transfers above 5,000 require a one-time password
- OTP expires after 300 seconds
