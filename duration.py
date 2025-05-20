from mutagen.mp3 import MP3

audio = MP3(r"C:\Users\m.rahman\audio3\audio3\data\stage1\1.mp3")
print(f"Duration: {audio.info.length:.3f} seconds")
