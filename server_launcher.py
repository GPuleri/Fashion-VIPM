from Updater import Updater
import os, sys, platform, subprocess

def fileparts(fn):
	(dirName, fileName) = os.path.split(fn)
	(fileBaseName, fileExtension) = os.path.splitext(fileName)
	return dirName, fileBaseName, fileExtension


def imageHandler(bot, message, chat_id, local_filename):
	print(local_filename)
	# send message to user
	bot.sendMessage(chat_id, "Hi, please wait until the image is ready")
	# set matlab command
	if 'Linux' in platform.system():
		matlab_cmd = '/usr/local/bin/matlab'
	else:
		matlab_cmd = '"C:\\Users\\pule\\anaconda3\\python\\python.exe"'
	# set command to start matlab script "edges.m"
	cur_dir = os.path.dirname(os.path.realpath(__file__))
	#cmd = matlab_cmd + " -nodesktop -nosplash -nodisplay -wait -r \"addpath(\'" + cur_dir + "\'); color(\'" + local_filename + "\'); quit\""
	cmd = "python main.py --query "+ local_filename
	#print(cmd)
	# lunch command
	subprocess.call(cmd,shell=True)
	# send back the manipulated image
	dirName, fileBaseName, fileExtension = fileparts(local_filename)
	new_fn = os.path.join(dirName, 'SIFT_result.jpg')
	deepResult= os.path.join(dirName, 'DeepMethod_result.jpg')
	bot.sendImage(chat_id, new_fn, "")
	bot.sendImage(chat_id, deepResult, "")



if __name__ == "__main__":
	bot_id = '1211627469:AAGsJvk6ZiAsePOKuBNFsDrmG3eZaU4veic'
	updater = Updater(bot_id)
	updater.setPhotoHandler(imageHandler)
	updater.start()