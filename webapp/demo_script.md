# AI Firefighter - 90-second demo script

Narration for the Milestone 4 demo screen capture. Spoken live while driving the
replay console. Bold is the action on screen, plain text is what is said.

Run the demo locally first:

    python3 -m http.server --directory webapp 8000
    # then open http://localhost:8000

---

**[0:00 - open on the console]**
"This is AI Firefighter. It converts a drone's wildfire feed into a structured recon brief for a CAL FIRE air attack supervisor."

**[0:12 - click the Sycan Marsh sequence in the left list]**
"This is the Sycan Marsh burn. The viewer is showing the actual drone frame, and I'll overlay the thermal hot area." **[toggle Thermal hot area on]**

**[0:24 - gesture at the readings, then click Begin analysis]**
"Each frame gets three independent readings: the DINOv2 detector, CLIP, and the 80-degree thermal measurement. Let me walk through how it decides."

**[0:34 - click Next step to d3, then to the gate]**
"First, d3 reads every frame and makes the initial call. Here its confidence lands at about 75 percent, below our 80 percent threshold, so it escalates for a second look."

**[0:50 - click Next step to the Qwen node; the tool calls stream in]**
"This is the tool-calling step. Qwen selects specific frames and invokes a 'decide' tool, asking about smoke, flames, and where the fire sits in the frame, then calls 'finish.' These are real tool calls, routed through liteLLM, our model harness, and they feed the written brief."

**[1:08 - step to the Final brief; the brief fills in on the right]**
"And here's the brief: fire confirmed, with its extent, its position in the frame, and the stated limitations."

**[1:16 - click a clear active-burn sequence (one without the escalated tag)]**
"Now compare a clear burn. d3 is confident on its own, so it bypasses the second model entirely. That's the fast path."

**[1:26 - hold on the screen]**
"The priority is latency. We take the fast path when the detector is confident and escalate only when it isn't, so the supervisor gets an accurate picture without spending compute we don't need."

---

Public demo video: https://www.youtube.com/watch?v=f_WylvfU87s
