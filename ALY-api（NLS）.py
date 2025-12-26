import time
import threading
import pyaudio
import nls
import json
import queue
import sys

# ==================== 配置 ====================
URL = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"#（地域选择--上海）
TOKEN = ""#（可通过阿里云SDK获取）
APPKEY = ""#(工作台用户里面-需要开启服务)

# 音频参数
CHUNK = 640
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


class RobustASR:
    """鲁棒的语音识别 """

    def __init__(self):
        self.transcriber = None
        self.is_connected = False
        self.is_recording = False
        self.result_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.start_event = threading.Event()
        self.error_event = threading.Event()
        self.keep_running = True

    def on_sentence_begin(self, message, *args):
        data = self._parse_message(message)
        data['event'] = 'sentence_begin'
        data['timestamp'] = time.time()
        self.result_queue.put(data)
        # print(f"[句子开始]")

    def on_sentence_end(self, message, *args):
        data = self._parse_message(message)
        data['event'] = 'sentence_end'
        data['timestamp'] = time.time()
        self.result_queue.put(data)

        if data.get('result'):
            print(f"\n识别结果: {data['result']}")
            if data.get('confidence', 0) > 0:
                print(f"   置信度: {data['confidence']:.1%}")

    def on_start(self, message, *args):
        data = self._parse_message(message)
        data['event'] = 'start'
        data['timestamp'] = time.time()
        self.result_queue.put(data)

        print("识别服务已启动")
        self.is_connected = True
        self.is_recording = True
        self.start_event.set()

    def on_result_changed(self, message, *args):
        data = self._parse_message(message)
        data['event'] = 'result_changed'
        data['timestamp'] = time.time()
        self.result_queue.put(data)

        if data.get('result') and len(data['result']) > 1:
            print(f"\r实时: {data['result']}", end="")

    def on_completed(self, message, *args):
        data = self._parse_message(message)
        data['event'] = 'completed'
        data['timestamp'] = time.time()
        self.result_queue.put(data)
        print("\n识别完成")
        self.is_recording = False

    def on_error(self, message, *args):
        data = {'event': 'error', 'message': message, 'timestamp': time.time()}
        self.result_queue.put(data)
        print(f"\n识别错误: {message}")
        self.is_recording = False
        self.error_event.set()

    def on_close(self, *args):
        data = {'event': 'close', 'timestamp': time.time()}
        self.result_queue.put(data)
        print("\n连接关闭")
        self.is_connected = False
        self.is_recording = False

    def _parse_message(self, message):
        """解析消息"""
        try:
            data = json.loads(message)
            header = data.get("header", {})
            payload = data.get("payload", {})

            return {
                'name': header.get("name", ""),
                'namespace': header.get("namespace", ""),
                'status': header.get("status", 0),
                'status_text': header.get("status_text", ""),
                'message_id': header.get("message_id", ""),
                'task_id': header.get("task_id", ""),
                'result': payload.get("result", ""),
                'confidence': payload.get("confidence", 0),
                'type': payload.get("type", ""),
                'time': payload.get("time", 0)
            }
        except:
            return {'raw': message[:100]}

    def start(self):
        """启动识别"""
        print("启动语音识别服务...")
        print(f"Token: {TOKEN[:20]}...{TOKEN[-20:]}")
        print(f"AppKey: {APPKEY}")

        # 创建语音识别器
        self.transcriber = nls.NlsSpeechTranscriber(
            url=URL,
            token=TOKEN,
            appkey=APPKEY,
            on_sentence_begin=self.on_sentence_begin,
            on_sentence_end=self.on_sentence_end,
            on_start=self.on_start,
            on_result_changed=self.on_result_changed,
            on_completed=self.on_completed,
            on_error=self.on_error,
            on_close=self.on_close
        )

        # 启动识别
        print("启动识别会话...")
        result = self.transcriber.start(
            aformat="pcm",#原始音频格式
            sample_rate=16000,
            enable_intermediate_result=True,
            enable_punctuation_prediction=True,
            enable_inverse_text_normalization=True
        )

        print(f"sr.start() 返回值: {result}")

        # 等待连接建立（最多5秒）
        print("等待连接建立...")
        if self.start_event.wait(timeout=5):
            print("连接建立成功")
            return True
        elif self.error_event.is_set():
            print("连接建立失败")
            return False
        else:
            # 没有收到start事件，但也没有错误
            print(" 未收到启动确认，但继续尝试...")
            return True

    def send_audio(self, audio_data):
        """发送音频数据"""
        if self.transcriber and self.is_recording:
            try:
                self.transcriber.send_audio(audio_data)
                return True
            except Exception as e:
                print(f"[发送音频错误] {e}")
                return False
        return False

    def get_result(self, timeout=0.1):
        """获取一个结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """停止识别"""
        self.keep_running = False
        self.is_recording = False

        if self.transcriber:
            try:
                self.transcriber.stop()
                time.sleep(0.5)
                print("识别已停止")
            except Exception as e:
                print(f"停止识别时出错: {e}")


def main():
    """主函数"""

    # 初始化ASR
    asr = RobustASR()

    # 启动识别
    if not asr.start():
        print("识别启动失败")
        return

    # 初始化音频设备
    print("\n初始化音频设备...")
    try:
        p = pyaudio.PyAudio()

        # 查找设备
        device_index = None
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                device_index = i
                print(f"使用设备: {info['name']}")
                break

        if device_index is None:
            print("无音频设备")
            p.terminate()
            asr.stop()
            return

        # 打开音频流
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK
        )

        print(f"音频参数: {CHANNELS}通道, {RATE}Hz")
        print("\n正在监听，请开始说话...\n")

        # 启动结果处理线程
        def process_results():
            """处理识别结果"""
            result_count = 0
            while asr.keep_running:
                result = asr.get_result(timeout=0.5)
                if result:
                    result_count += 1
                    # 这里可以处理结果
                    # print(f"[结果{result_count}] 事件: {result.get('event')}")
                time.sleep(0.01)

        result_thread = threading.Thread(target=process_results)
        result_thread.daemon = True
        result_thread.start()

        data_count = 0
        start_time = time.time()

        try:
            while asr.keep_running and asr.is_recording:
                # 读取音频
                audio_data = stream.read(CHUNK, exception_on_overflow=False)
                data_count += 1

                # 发送音频
                asr.send_audio(audio_data)

                # 显示状态
                if data_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"\r时长: {elapsed:.1f}秒 | 数据包: {data_count}", end="")

                time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\n[用户中断] 停止识别...")
        except Exception as e:
            print(f"\n[错误] {e}")
        finally:
            # 清理资源
            print("\n[清理] 释放资源...")
            asr.keep_running = False
            stream.stop_stream()
            stream.close()
            p.terminate()
            asr.stop()

            print("\n完成")

    except Exception as e:
        print(f"初始化失败: {e}")
        asr.stop()


if __name__ == "__main__":
    nls.enableTrace(False)
    main()
