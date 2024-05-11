import sounddevice as sd
import numpy as np
import librosa
import multiprocessing
import time
import cv2

def audio_stream(queue):
    fs = 44100  # Sample rate
    duration = 30  # Duration of chunks in seconds
    
    def callback(indata, frames, time, status):
        if status:
            print(status)
        queue.put(indata.copy())  # Put the chunk into the queue for processing

    with sd.InputStream(samplerate=fs, channels=1, callback=callback):
        print("Streaming started...")

        

def feature_extraction(queue):
    # Initialize variables for continuous feature extraction
    fs = 44100  # Sample rate
    hop_length = 512  # Hop length for feature extraction
    
    while True:
        audio_chunk = queue.get()
        if audio_chunk is not None:
            # Convert the audio chunk to floating-point values
            audio_chunk = audio_chunk.astype(np.float32)
            
            # Normalize the audio chunk
            max_abs = np.max(np.abs(audio_chunk))
            if max_abs > 0:
                audio_chunk /= max_abs
            
            print("Processing a new audio chunk...")
            
            # Extract features from the entire audio chunk
            rms_energy = np.mean(librosa.feature.rms(y=audio_chunk, hop_length=hop_length))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_chunk, sr=fs, hop_length=hop_length))
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_chunk, hop_length=hop_length))

            features = {
                'RMS Energy': rms_energy,
                'Spectral Centroid': spectral_centroid,
                'Zero Crossing Rate': zcr,
            }
            print("Extracted Features:", features)


def video_capture(queue):
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        print("Error: camera is not opened")
        return
    try:
        while True:
            ret, frame = videocapture.read()
            if not ret:
                break
            queue.put(frame)
    finally:
        videocapture.release()

def stream(queue):
    cv2.namedWindow("Live Capture", cv2.WINDOW_NORMAL)
    while True:
        frame = queue.get()
        if frame is None:
            continue
        cv2.imshow("Live Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def main():
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue(maxsize=10)

    process_audio = ctx.Process(target=audio_stream, args=(queue,))
    process_features = ctx.Process(target=feature_extraction, args=(queue,))
    process_capture = ctx.Process(target=video_capture, args=(queue,))
    process_stream = ctx.Process(target=stream, args=(queue,))
    process_audio.start()
    process_features.start()
    process_capture.start()
    process_stream.start()

    try:
        process_audio.join()
        process_features.join()
        process_capture.join()
        process_stream.join()
    except KeyboardInterrupt:
        print("Terminating processes...")
        process_audio.terminate()
        process_features.terminate()
        process_capture.terminate()
        process_stream.terminate()

    print("All processes terminated.")

if __name__ == "__main__":
    main()
