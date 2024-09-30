# Seamless Streaming with QUIC

## Getting Setup

* I am probably missing some steps

### Server

1. Deploy to GCP

* Set the values in `config.sh`
* run `sh create.sh`
* run `sh ports.sh` // will open the quic ports
* run `sh ip.sh` // will print the ip address of the server
* run `sh ssh.sh` // ssh into the server
* run `sh stop.sh` // to stop the server when you are done so you dont go broke

2. Once sshed into the server, `cd app` `python server.py`
  You will have to pip install stuff

### Client

1. Set the IP address of the server in `app/client.py`
2. `python app/client.py`

Trying to debug a couple issues.

* Testing with a hardcoded spanish audio file that says "how are you"

1. I get this weird translation initially "all of it, all of it, all of it, all of it, all of it". It seems to just repeat the same text over and over again.

```
2024-09-30 18:34:19,130 INFO -- server: Connection made.
2024-09-30 18:34:19,132 INFO -- quic: [066b4886080a7db7] Negotiated protocol version 0x00000001 (VERSION_1)
2024-09-30 18:34:21,248 INFO -- server: [send_output] Got text segment: All of it, all
2024-09-30 18:34:21,248 INFO -- server: [send_output] Got speech segment
2024-09-30 18:34:23,104 INFO -- server: [send_output] Got text segment: of it, all of it, all of it, all of it, all of it.
2024-09-30 18:34:23,104 INFO -- server: [send_output] Got speech segment
2024-09-30 18:34:26,575 INFO -- server: [send_output] Got text segment: How
2024-09-30 18:34:26,576 WARNING -- server: [send_output] Received non-speech segment.
2024-09-30 18:34:26,739 INFO -- server: Connection terminated
2024-09-30 18:34:26,739 INFO -- server: Cancelling tasks
2024-09-30 18:34:26,740 INFO -- server: [send_output] Sender task cancelled
2024-09-30 18:34:28,468 INFO -- server: Connection made.
2024-09-30 18:34:28,469 INFO -- quic: [2618da09a7676177] Negotiated protocol version 0x00000001 (VERSION_1)
2024-09-30 18:34:28,857 INFO -- server: [send_output] Got text segment: are
2024-09-30 18:34:28,857 WARNING -- server: [send_output] Received non-speech segment.
2024-09-30 18:34:31,501 INFO -- server: [send_output] Got text segment: you
2024-09-30 18:34:31,501 WARNING -- server: [send_output] Received non-speech segment.
2024-09-30 18:34:33,224 INFO -- server: [send_output] Got text segment: ?
2024-09-30 18:34:33,224 WARNING -- server: [send_output] Received non-speech segment.
2024-09-30 18:34:35,365 INFO -- server: [send_output] Got text segment: How
2024-09-30 18:34:35,365 WARNING -- server: [send_output] Received non-speech segment.
2024-09-30 18:34:39,059 INFO -- server: [send_output] Got text segment: are you?
2024-09-30 18:34:39,059 INFO -- server: [send_output] Got speech segment
```

2. When I do finally get the correct translation, it also seems to be duplicated





## References

* https://github.com/facebookresearch/seamless_communication/blob/main/Seamless_Tutorial.ipynb
* https://huggingface.co/spaces/facebook/seamless-streaming/tree/main