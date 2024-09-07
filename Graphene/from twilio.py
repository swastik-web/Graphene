from twilio.rest import Client

# Your Twilio credentials
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
twilio_phone_number = 'your_twilio_phone_number'

# Create a Twilio client
client = Client(account_sid, auth_token)

# Send an SMS
message = client.messages.create(
    body="Hello, this is a test message!",
    from_=twilio_phone_number,
    to='+1234567890'  # Replace with the recipient's phone number
)

# Print the message SID (a unique identifier for this message)
print(f"Message sent with SID: {message.sid}")
