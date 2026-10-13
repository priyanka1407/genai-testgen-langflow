# Release Notes - Core Banking v4.2

## Payments

### Domestic fund transfer daily limit
- A retail customer can transfer up to 10,000 USD per calendar day across all domestic transfers
- A transfer that would exceed the daily limit is rejected with error code PAY-LIMIT-001
- The daily limit resets at 00:00 in the customer's home timezone

### Transfer amount validation
- Minimum transfer amount is 1.00
- Amount must have at most 2 decimal places
- Source and destination account cannot be the same

## Accounts

### Account status checks
- Transfers from accounts in status FROZEN or CLOSED are rejected
- A DORMANT account must be reactivated before any debit
