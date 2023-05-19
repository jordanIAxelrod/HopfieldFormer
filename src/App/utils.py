def format_message(messages):
    return_string = []
    for message in messages:
        return_string.append(f'<|{message["role"]}|> {message["text"]}\n\n')
    return ''.join(return_string)