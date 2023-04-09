from GPTHandler import GPTHandler
import retroBot
from retroBot.config import config as GPTConfig
import re
import logging
import logging.handlers
import os
import openai

class GPTBot(retroBot.retroBot):

    def __init__(self, config):
        self.config = config
        openai.api_key = config['gpt']['api_key']
        #self.blacklist_file = config['gpt']['blacklist_file']
        #self.blacklist_words = self.load_blacklist(self.blacklist_file)
        self.username = config['twitch']['username']
        self.client_id = config['twitch']['client_id']
        self.client_secret = config['twitch']['client_secret']
        for channel in config['twitch']['channels']:
            channel_config = config['twitch']['channels'][channel]
            for setting in config['gpt']['defaults']:
                if not setting in channel_config or not channel_config[setting]:
                    channel_config[setting] = config['gpt']['defaults'][setting]
            self.config.save()
        super(GPTBot, self).__init__(
            config['twitch']['username'],
            config['twitch']['client_id'],
            config['twitch']['client_secret'],
            config['twitch']['channels'],
            handler=GPTHandler
        )
        
    def load_blacklist(self, blacklist_file):
        with open(blacklist_file, 'r') as f:
            words = [line.rstrip('\n') for line in f]
        return words

    def checkBlacklisted(self, message):
        # Check words that the bot should NEVER learn.
        for i in self.blacklist_words:
            if re.search(r"\b" + i, message, re.IGNORECASE):
                return True
        return False


def main():
    logger = setup_logger('retroBot')
    config = load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    bot = GPTBot(config)
    bot.start()

def load_config(filename):
    config = GPTConfig(filename)
    config.save()
    return config

def setup_logger(logname, logpath=""):
    if not logpath or logpath == "":
        logpath = os.path.join(os.path.dirname(__file__), 'logs')
    else:
        logpath = os.path.abspath(logpath)
    if not os.path.exists(logpath):
        os.mkdir(logpath)
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.handlers.TimedRotatingFileHandler(os.path.join(logpath, logname), when='midnight')
    stream_handler = logging.StreamHandler()
    form = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(form)
    stream_handler.setFormatter(form)
    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

if __name__ == '__main__':
    main()