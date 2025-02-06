import logging
import socket
import sys
import os
# get ready to listen on port 11199
PORT=11199
#10kb buffer should be big enough for most audio chunks
BUFFER_SIZE=4096

AVAILABLE_MODELS = ["tiny.en", "base.en", "medium.en", "large-v3"]
MODEL_IDX = 3

# are we collecting logs?
LOGGING = True
def start_logging(LOGGING) :
    if (LOGGING) :
        if (not os.path.isdir("logs")) : os.mkdir("logs")
        logging.basicConfig(filename="logs/log.txt", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def set_trigger_word() :
    # what word triggers out agent to act?
    try :
        trigger_word = open("trigger_word.txt", "rt").readline()
    except OSError :
        # default trigger word creates a file we can later edit
        trigger_word = "jarvis"
        trigger_word_file = open("trigger_word.txt", "xt")
        trigger_word_file.write(trigger_word)
        trigger_word_file.close()
    return trigger_word


def get_model_selection() :
    # different options for whisper models
    # TODO when we hit a RuntimeError: CUDA out of memory we should try to fail back to a smaller model
    # WHISPERMODEL = "tiny.en"
    # WHISPERMODEL = "base.en"
    # WHISPERMODEL = "medium.en"
    if (len(sys.argv) == 1) :
        whisper_model = AVAILABLE_MODELS[MODEL_IDX]
    else :
        if (sys.argv[1] in AVAILABLE_MODELS) :
            MODEL_IDX = AVAILABLE_MODELS.index(sys.argv[1])
            whisper_model = AVAILABLE_MODELS[MODEL_IDX]
        else :
            if (LOGGING) : logging.debug(f"Invalid choice of model, try again with one of the following: {AVAILABLE_MODELS}")
    if (LOGGING) : logging.debug(f"Model Selected: {whisper_model}")
    return whisper_model

    
def load_model(whisper_model) :
    from faster_whisper import WhisperModel
    if (LOGGING) : logging.debug("Loading Model")
    try :
        voicemodel = WhisperModel(whisper_model, device="cuda",compute_type="float16")
        if (LOGGING) : logging.debug("Successfully loaded model")
    except ValueError :
        try :
            voicemodel = WhisperModel(whisper_model, device="cuda",compute_type="int8")
            if (LOGGING) : logging.debug("Failed to load model, retrying with less precision")
        except ValueError :
            voicemodel = WhisperModel(whisper_model, device="cpu",compute_type="int8")
            if (LOGGING) : logging.debug("Failed to utilize GPU, falling back to CPU operation")
    return voicemodel


def connect_to_client() :
    if (LOGGING) : logging.debug("Waiting for the client to connect")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket :
        client_socket.bind(("",PORT))
        client_socket.listen(1)
        try :
            return client_socket.accept()
        except :
            if (LOGGING) : logging.debug("Failed to reconnect to client, going back to listening")


def get_client_data(client_connection, ip_address) :
    if (LOGGING) : logging.debug("Waiting on data from the client")
    # Need to block for the first packet then recieve the rest of the batch
    # TODO This will get complicated if we recieve more data before we're done processing
    encoded_voice_data = b""
    # Wait for the first packet indefinetly
    voice_data_packet = client_connection.recv(BUFFER_SIZE)
    # Then wait up to 0.5 seconds for each packet to make sure we collect all the data
    encoded_voice_data += voice_data_packet
    client_connection.settimeout(0.5)
    while True :
        try :
            voice_data_packet = client_connection.recv(BUFFER_SIZE)
            if (voice_data_packet == b"") :
                break
        except BlockingIOError :
            break
        except TimeoutError :
            break
        encoded_voice_data += voice_data_packet
    client_connection.settimeout(None)

    if (voice_data_packet == b"") :
        if (LOGGING) : logging.debug(f"Connection closed by client {ip_address[0]}, shutting down")
        raise Exception("Client connection closed")

    if (LOGGING) : logging.debug(f"Recieved Packet of Size: {sys.getsizeof(encoded_voice_data)} bytes from {ip_address[0]}")    
    try :
        import numpy as np
        voice_data = encoded_voice_data.decode()
        voice_data_cleaned = voice_data.strip()[1:-1]
        voice_data_array = np.fromstring(voice_data_cleaned, sep=",", dtype=np.float32)
        return voice_data_array
    except EOFError :
        if (LOGGING) : logging.debug(f"Connection closed by client {ip_address[0]}, shutting down")
        raise Exception("Client connection closed")


def transmit_data(client_connection, text) :
    text_packet = text.encode()
    client_connection.sendall(text_packet)


def process_data(voicemodel, voice_data_array, trigger_word) :
    segments, _ = voicemodel.transcribe(voice_data_array,language="en",beam_size=5,no_speech_threshold=0.33,initial_prompt=f"{trigger_word} open some app")
    segments = list(segments)
    text = ""
    for s in segments :
        text+=s.text
    return text

def select_smaller_model() :
    if (MODEL_IDX > 0) :
        MODEL_IDX -= 1
        if (LOGGING) : logging.debug(f"Ran out of VRAM, scaling down to a smaller model")
    else :
        if (LOGGING) : logging.debug(f"Not enough VRAM to use Whisper reliably, or there is a memory leak")
        raise Exception("Ran out of VRAM")

def main() :
    start_logging(LOGGING)
    trigger_word = set_trigger_word()
    while True :
        whisper_model = get_model_selection()
        voice_model = load_model(whisper_model)
        try :
            while True :
                client_connection, ip_address = connect_to_client()
                with client_connection :
                    while True :
                        try :
                            client_data = get_client_data(client_connection, ip_address)
                        except :
                            # If the client connection breaks, reconnect
                            break
                        text = process_data(voice_model, client_data, trigger_word)
                        transmit_data(client_connection, text)
        except RuntimeError :
            select_smaller_model()     

if __name__ == "__main__" :
    main()