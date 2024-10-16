#!/usr/bin/env python3
## python recordtest.py out.raw # talk to the microphone
## aplay -r 8000 -f S16_LE -c 1 out.raw
#!/usr/bin/env python

from __future__ import print_function

import sys
import time
import getopt
import alsaaudio
import wave
import speech_recognition as sr
import _thread

###########################################################################################################

def usage():
	print('usage: recordtest.py [-d <device>] <file>', file=sys.stderr)
	sys.exit(2)
	
def play(device, f):	
	format = None
	# 8bit is unsigned in wav files
	if f.getsampwidth() == 1:
		format = alsaaudio.PCM_FORMAT_U8
	# Otherwise we assume signed data, little endian
	elif f.getsampwidth() == 2:
		format = alsaaudio.PCM_FORMAT_S16_LE
	elif f.getsampwidth() == 3:
		format = alsaaudio.PCM_FORMAT_S24_3LE
	elif f.getsampwidth() == 4:
		format = alsaaudio.PCM_FORMAT_S32_LE
	else:
		raise ValueError('Unsupported format')

	periodsize = f.getframerate() // 8
	#print('%d channels, %d sampling rate, format %d, periodsize %d\n', (f.getnchannels(), f.getframerate(), format, periodsize)) 
	device = alsaaudio.PCM(channels=f.getnchannels(), rate=f.getframerate(), format=format, periodsize=periodsize, device=device)
	data = f.readframes(periodsize)
	while data:
		# Read data from stdin
		if device.write(data) < 0:
			print("Playback buffer underrun! Continuing nonetheless ...")
		data = f.readframes(periodsize)	

def input_thread(a_list):
    input()            
    a_list.append(True)

###########################################################################################################

if __name__ == '__main__':

	device = 'default'

	opts, args = getopt.getopt(sys.argv[1:], 'd:')
	for o, a in opts:
		if o == '-d':
			device = a

	if not args:
		usage()

	f = open(args[0], 'wb')

	# Open the device in nonblocking capture mode in mono, with a sampling rate of 44100 Hz
	# and 16 bit little endian samples
	# The period size controls the internal number of frames per period.
	# The significance of this parameter is documented in the ALSA api.
	# For our purposes, it is suficcient to know that reads from the device
	# will return this many frames. Each frame being 2 bytes long.
	# This means that the reads below will return either 320 bytes of data
	# or 0 bytes of data. The latter is possible because we are in nonblocking
	# mode.
	inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK,
		channels=1, rate=44100, format=alsaaudio.PCM_FORMAT_S16_LE,
		periodsize=160, device=device)
	input("please press any key to start recording, press 'q' to stop recording")
	a_list = []
	_thread.start_new_thread(input_thread, (a_list,))
	while not a_list: #True:
		# Read data from device
		l, data = inp.read()
		if l < 0:
			print("Capture buffer overrun! Continuing nonetheless ...")
		elif l:
			f.write(data)
			time.sleep(.001)
				
	inname = args[0]
	rawname = inname[0:len(inname)-4]
	
###########################################################################################################

	with open(args[0], "rb") as inp_f:
		data = inp_f.read()
		with wave.open(rawname+".wav", "wb") as out_f:
			out_f.setnchannels(1)
			out_f.setsampwidth(2) # number of bytes
			out_f.setframerate(44100)
			out_f.writeframesraw(data)
        
###########################################################################################################
	with wave.open(rawname+".wav", 'rb') as fx:
		play(device, fx)
###########################################################################################################		
		
	r = sr.Recognizer()
	hellow=sr.AudioFile(rawname+".wav")
	with hellow as source:
		audio = r.record(source)
		try:
			s = r.recognize_google(audio)
			print("Text: "+s)
		except Exception as e:
			print("ExceptionX: "+str(e))
