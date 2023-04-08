import openai
from openai.upload_progress import BufferReader
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
        self.initMessageDB(cursor)
        self.initConfigDB(cursor)
        self.initModelDB(cursor)
        connection.commit()
        cursor.close()
        connection.close()

    def initMessageDB(self, cursor):
        cursor.execute('create table if not exists messages(message TEXT NOT NULL)')
    
    def initConfigDB(self, cursor):
        cursor.execute('create table if not exists config(key TEXT NOT NULL, value TEXT)')

    def initModelDB(self, cursor):
        cursor.execute('create table if not exists models(iteration INTEGER NOT NULL PRIMARY KEY, date TIMESTAMP NOT NULL, message_count INTEGER NOT NULL, model TEXT NOT NULL)')
    
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
    
    def generateMessage(self, **kwargs):
        generator = openai.Completion.create(model=self.model, max_tokens=self.max_tokens, stop=['\n'], **kwargs)
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

    def writeMessage(self, msg):
        message = self.filterMessage(msg.content)
        if message:
            connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
            cursor = connection.cursor()
            cursor.execute('insert into messages values (?)', (message))
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

    def fineTuneModel(self, poll_interval=POLL_INTERVAL, **kwargs):

        dataset = self.retrieveDataSet()
        cutoff_row = dataset[-1][0]
        dataset_length = len(dataset)
        dataset = self.formatDataSet(dataset)
        dataset = self.uploadDataSet(dataset)

        resp = openai.FineTune.create(training_file=dataset, **kwargs)
        job_id = resp['id']
        self.logger.info(f"Created fine-tuning job: {job_id}")
        self.logger.debug(resp)

        status = 'pending'
        while status == 'pending':
            time.sleep(poll_interval)
            resp = openai.FineTune.retrieve(id=job_id)
            status = resp['status']
            self.logger.debug(resp)
        
        finished = False
        events = openai.FineTune.stream_events(job_id)
        while finished == False:
            try:
                for event in events:
                    self.logger.info(
                        "[%s] %s"
                        % (
                            datetime.datetime.fromtimestamp(event["created_at"]),
                            event["message"],
                        )
                    )
                finished = True
            except Exception as e:
                self.logger.warning(f'Experienced the following exception while waiting for fine tuning to finish. Will retry: {e}')
                events = None
                while events == None:
                    try:
                        resp = openai.FineTune.retrieve(id=job_id)
                        if resp["status"] in ['pending', 'running']:
                            events = openai.FineTune.stream_events(job_id)
                        else:
                            finished = True
                    except Exception as e:
                        self.logger.warning(f'Experienced the following exception while retrying connection to fine tuning job. Will retry: {e}')

        resp = openai.FineTune.retrieve(id=job_id)
        self.logger.debug(resp)

        if resp["status"] == "succeeded":
            self.logger.info(f'Fine tuning model creation has succeeded! The resulting model is: {resp["fine_tuned_model"]}')
            created_date = datetime.datetime.fromtimestamp(resp["result_files"]["created_at"])
            self.setModel(resp["fine_tuned_model"], dataset_length, created_date)
            self.pruneMessages(cutoff_row)
            return resp["fine_tuned_model"]
        elif resp["status"] == "failed":
            self.logger.error(f'Fine tuning model creation has failed!')
            return None
        else:
            self.logger.error(f'Fine tuning model creation has exploded! My god!')
            return None

    def setModel(self, model, dataset_length, created_date):
        self.model = model
        connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
        cursor = connection.cursor()
        cursor.execute('insert into models values (?, ?, ?, ?)', (None, created_date, dataset_length, model))
        connection.commit()
        cursor.close()
        connection.close()

    def retrieveDataSet(self):
        connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
        cursor = connection.cursor()
        cursor.execute('select rowid, message from messages')
        rows = cursor.fetchall()
        cursor.close()
        connection.close()
        self.logger.debug(f'Retrieved rows {rows[0][0]} through {rows[-1][0]}')
        return rows
    
    def formatDataSet(self, dataset):
        jsonl_content = ""
        for row in dataset:
            jsonl_content += '{"prompt": "\n", "completion": "' + row[1] + '"}\n'
        return jsonl_content
    
    def uploadDataSet(self, dataset):
        file_name = f"{self.channel}_{time.time()}"
        buffer_reader = BufferReader(dataset, desc="Upload progress")
        resp = openai.File.create(
            file=buffer_reader,
            purpose="fine-tune",
            user_provided_filename=file_name
        )
        self.logger.debug(
            "Uploaded file from {file}: {id}".format(
                file=file_name, id=resp["id"]
            )
        )
        return resp["id"]

    def pruneMessages(self, cutoff_row):
        connection = sqlite3.connect(self.db_file, timeout=self.db_timeout)
        cursor = connection.cursor()
        cursor.execute('delete from messages where rowid <= ?', (cutoff_row,))
        connection.commit()
        cursor.execute('vacuum')
        connection.commit()
        cursor.close()
        connection.close()
    
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