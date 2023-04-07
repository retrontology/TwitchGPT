import openai
import retroBot.channelHandler
from retroBot.message import message
import os
import re
import datetime
import sqlite3
import time

POLL_INTERVAL=5

class GPTHandler(retroBot.channelHandler):

    def __init__(self, channel, parent, *args, **kwargs):
        super().__init__(channel, parent)
        self.user_id = parent.twitch.get_users(logins=[channel.lower()])['data'][0]['id']
        self.message_count = 0
        self.initDB()
        self.last_train = datetime.datetime.now()
        self.model = parent.config['twitch']['channels'][channel]['model']
        self.max_tokens = parent.config['twitch']['channels'][channel]['max_tokens']
        self.send_messages = parent.config['twitch']['channels'][channel]['send_messages']
        self.generate_on = parent.config['twitch']['channels'][channel]['generate_on']
        self.ignored_users = [x.lower() for x in self.parent.config['twitch']['channels'][channel]['ignored_users']]
        self.initCooldowns()
        
    def initCooldowns(self):
        self.cooldowns = {}
        self.last_used = {}
        self.cooldowns['speak'] = 300
        self.last_used['speak'] = datetime.datetime.fromtimestamp(0)
        self.cooldowns['commands'] = 300
        self.last_used['commands'] = datetime.datetime.fromtimestamp(0)
        self.cooldowns['reply'] = 120
        self.last_used['reply'] = datetime.datetime.fromtimestamp(0)

    def initDB(self):
        self.db_timeout = 10
        dir = os.path.join(os.path.dirname(__file__), 'messages')
        if not os.path.isdir(dir): os.mkdir(dir)
        self.db_file = os.path.join(dir, f'{self.channel.lower()}.db')
        connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
        sqlite3.register_adapter(bool, int)
        sqlite3.register_converter("BOOLEAN", lambda v: bool(int(v)))
        cursor = connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        connection.commit()
        cursor.close()
        connection.close()
        self.initMessageDB()
        self.initConfigDB()
        self.initModelDB()

    def initMessageDB(self):
        connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
        cursor = connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('create table if not exists messages(date timestamp, user_id integer, name text, mod BOOLEAN, message text)')
        connection.commit()
        cursor.close()
        connection.close()
    
    def initConfigDB(self):
        connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
        cursor = connection.cursor()
        cursor.execute('create table if not exists config(key text, value text)')
        connection.commit()
        cursor.close()
        connection.close()

    def initModelDB(self):
        connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
        cursor = connection.cursor()
        cursor.execute('create table if not exists models(iteration integer, date timestamp, message_count integer, model text)')
        connection.commit()
        cursor.close()
        connection.close()
    
    def on_pubmsg(self, c, e):
        msg = message(e)
        if msg.username.lower() in self.ignored_users:
            return
        elif msg.content[:1] == '!':
            self.handleCommands(msg)
        elif msg.content.lower().find(f'@{self.parent.username.lower()}') != -1:
            self.logger.info(f'{msg.username}: {msg.content}')
            if (datetime.datetime.now() - self.last_used['reply']).total_seconds() >= self.cooldowns['reply']:
                self.generateAndSendMessage(msg.username)
                self.last_used['reply'] = datetime.datetime.now()
        else:
            self.writeMessage(msg)
        if self.message_count >= self.generate_on:
            self.generateAndSendMessage()
    
    def generateMessage(self):
        generator = openai.Completion.create(model=self.model, max_tokens=self.max_tokens, stop=['\n'])
        return generator.choices[0].text

    def generateAndSendMessage(self, target=None):
        try:
            self.message_count = 0
            generated = self.generateMessage()
        except Exception as e:
            self.logger.error(e)
            generated = None
        if generated != None:
            if target != None:
                generated = f'@{target} {generated}'
            self.logger.info(f'Generated: {generated}')
            if self.send_messages: self.send_message(generated)
        else:
            self.logger.error("Could not generate a message :(")
        self.checkCull()

    def writeMessage(self, msg):
        message = self.filterMessage(msg.content)
        if message:
            connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
            cursor = connection.cursor()
            cursor.execute('insert into messages values (?, ?, ?, ?, ?)', (msg.time, msg.user_id, msg.username, msg.mod, message))
            connection.commit()
            cursor.close()
            connection.close()
            self.message_count += 1
            return True
        else:
            return False
    
    def filterMessage(self, message):
        if self.parent.checkBlacklisted(message):
            return False
        # Remove links
        # TODO: Fix
        message = re.sub(r"http\S+", "", message)
        # Remove mentions
        if self.parent.allow_mentions == False:
            message = re.sub(r"@\S+", "", message)
        # Remove just repeated messages.
        words = message.split()
        # Space filtering
        message = re.sub(r" +", " ", message)
        message = message.strip()
        return message

    def fineTuneModel(self, poll_interval=POLL_INTERVAL):
        dataset_file = self.prepareDataSet()
        create_args = {"training_file": dataset_file}
        resp = openai.FineTune.create(**create_args)
        job_id = resp['id']

        self.logger.info(f"Created fine-tuning job: {job_id}")
        self.logger.debug(resp)

        status = 'pending'
        while status == 'pending':
            time.sleep(poll_interval)
            resp = openai.FineTune.retrieve(id=job_id)
            status = resp['status']
            self.logger.debug(resp)
        
        events = openai.FineTune.stream_events(job_id)
        try:
            for event in events:
                self.logger.info(
                    "[%s] %s"
                    % (
                        datetime.datetime.fromtimestamp(event["created_at"]),
                        event["message"],
                    )
                )
        except Exception as e:
            self.logger.warning(f'Experienced the following exception while waiting for fine tuning to finish. Will retry: {e}')
            events = None
            while events == None:
                try:
                    resp = openai.FineTune.retrieve(id=job_id)
                    if resp["status"] in ['pending', 'running']:
                        events = openai.FineTune.stream_events(job_id)
                    else:
                        break
                except Exception as e:
                    self.logger.warning(f'Experienced the following exception while retrying connection to fine tuning job. Will retry: {e}')

        resp = openai.FineTune.retrieve(id=job_id)
        self.logger.debug(resp)

        #bookmark TODO
        if resp["status"] == "succeeded":
            self.logger.info(f'Fine tuning model creation has succeeded! The resulting model is: {resp["fine_tuned_model"]}')
            return resp["fine_tuned_model"]
        elif resp["status"] == "failed":
            self.logger.error(f'Fine tuning model creation has failed!')
            return None
        return resp["id"]

    def prepareDataSet(self):
        connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
        cursor = connection.cursor()
        return "somedatasetfilename"

    def cullFile(self):
        connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
        cursor = connection.cursor()
        size = cursor.execute('select count(*) from messages').fetchall()[0][0]
        self.logger.debug(f'Size of messages: {size}')
        if size > self.parent.cull_over:
            size_delete = size // 2
            self.logger.debug(f'Culling rows below: {size_delete}')
            cursor.execute('delete from messages where rowid < ?', (size_delete,))
            connection.commit()
            cursor.execute('vacuum')
        cursor.close()
        connection.close()
    
    def checkCull(self):
        now_time = datetime.datetime.now()
        time_since_cull = now_time - self.last_cull
        self.logger.debug(f'Time since last cull: {time_since_cull.total_seconds()}')
        if time_since_cull.total_seconds() > self.parent.time_to_cull:
            self.cullFile()
            self.last_cull = datetime.datetime.now()
    
    def handleCommands(self, msg):
        cmd = msg.content.split(' ')[0][1:].lower()
        if cmd == 'commands' and (datetime.datetime.now() - self.last_used[cmd]).total_seconds() >= self.cooldowns[cmd]:
            self.send_message('You can find a list of my commands here: https://www.retrontology.com/index.php/neuralbronson-commands/')
            self.last_used[cmd] = datetime.datetime.now()
        elif cmd == 'speak' and (datetime.datetime.now() - self.last_used[cmd]).total_seconds() >= self.cooldowns[cmd]:
            self.generateAndSendMessage()
            self.last_used[cmd] = datetime.datetime.now()
        if msg.mod or msg.broadcaster or msg.user_id in [54714257, 37749713]:
            if cmd == 'clear':
                if self.clear_logs_after:
                    self.clear_logs_after = False
                    self.parent.config.save()
                    self.send_message("No longer clearing memory after message! MrDestructoid")
                else:
                    self.clear_logs_after = True
                    self.parent.config.save()
                    self.send_message("Clearing memory after every message! MrDestructoid")
            elif cmd == 'wipe':
                connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
                cursor = connection.cursor()
                cursor.execute('delete from messages')
                connection.commit()
                cursor.execute('vacuum')
                cursor.close()
                connection.close()
                self.send_message("Wiped memory banks. MrDestructoid")
            elif cmd == 'toggle':
                if self.send_messages:
                    self.send_messages = False
                    self.parent.config.save()
                    self.send_message("Messages will no longer be sent! MrDestructoid")
                else:
                    self.send_messages = True
                    self.parent.config.save()
                    self.send_message("Messages are now turned on! MrDestructoid")
            elif cmd == 'unique':
                if self.unique:
                    self.unique = False
                    self.parent.config.save()
                    self.send_message("Messages will no longer be unique. MrDestructoid")
                else:
                    self.unique = True
                    self.parent.config.save()
                    self.send_message("Messages will now be unique. MrDestructoid")
            elif cmd == 'setafter':
                try:
                    stringNum = msg.content.split(' ')[1]
                    if stringNum != None:
                        num = int(stringNum)
                        if num <= 0:
                            raise Exception
                        self.generate_on = num
                        self.parent.config.save()
                        self.send_message("Messages will now be sent after " + self.generate_on + " chat messages. MrDestructoid")
                except:
                        self.send_message("Current value: " + str(self.generate_on) + ". To set, use: setafter [number of messages]")
            elif cmd == 'isalive':
                self.send_message("Yeah, I'm alive and learning. MrDestructoid")