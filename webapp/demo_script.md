# AI Firefighter - 90-second demo script

Narration for the Milestone 4 demo screen capture. Spoken live while driving the
replay console. Bold is the action on screen, plain text is what is said.

Run the demo locally first:

    python3 -m http.server --directory webapp 8000
    # then open http://localhost:8000

---

**[0:00 - open on the console]**
"This is AI Firefighter. It turns a drone's wildfire feed into a structured recon brief for a CAL FIRE air attack supervisor. It's decision support with a human in the loop. It never dispatches crews."

**[0:12 - click the Sycan Marsh sequence in the left list]**
"Here's the Sycan Marsh burn, tagged 'escalated' on the left. The viewer is showing the real drone frame, and I'll turn on the thermal hot area." **[toggle Thermal hot area on]**

**[0:24 - gesture at the readings, then click Begin analysis]**
"Every frame gets three readings: the DINOv2 detector, CLIP, and the 80-degree thermal measurement. Let me step through how it decides."

**[0:34 - click Next step to d3, then to the gate]**
"First, d3 reads all the frames and makes the call. Here it's only about 75 percent sure, which is under our 80 percent bar, so it escalates for a closer look."

**[0:50 - click Next step to the Qwen node; the tool calls stream in]**
"This is the tool-calling step. Qwen picks specific frames and calls a 'decide' tool, asking about smoke, flames, and where the fire is, then calls 'finish.' These are real tool calls, run through liteLLM, and they feed the written brief."

**[1:08 - step to the Final brief; the brief fills in on the right]**
"And here's the brief: fire confirmed, with its extent, its location in the frame, and the limitations."

**[1:16 - click a clear active-burn sequence (one without the escalated tag)]**
"Now compare a clear burn. d3 is confident on its own, so it skips the second model entirely. That's the fast path."

**[1:26 - hold on the screen]**
"The point is speed: we go fast when we can, and only escalate when the model isn't sure, because a fire is burning and the supervisor needs the picture now."

---

About 218 words, ~90 seconds at a natural pace. If a take runs long, cut the
opening sentence or the closing line first.

Public demo video: upload the recording to YouTube (unlisted is fine) and link it
from this repo and the submission PDF.
