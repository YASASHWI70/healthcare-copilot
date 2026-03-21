# 🧪 Example Prompts for Testing MediCopilot

Use these prompts to thoroughly test all aspects of the system.

---

## 🟢 LOW RISK Scenarios

### Common Cold
```
I have a runny nose, sneezing, mild sore throat, and slight fatigue.
Started yesterday. No fever. I feel generally under the weather.
```

### Mild Headache
```
I have a mild headache at the top of my head, feels like a band of pressure.
Been there for a few hours. I've been staring at a screen all day.
```

### Minor GI Upset
```
I have some nausea and mild stomach cramps after eating out last night.
No vomiting, no fever, just some discomfort.
```

---

## 🟡 MODERATE RISK Scenarios

### Flu-Like Illness
```
I've had a high fever of 38.8°C for 2 days along with severe body aches,
fatigue, dry cough, and chills. The fever isn't responding well to paracetamol.
```

### Possible COVID
```
I've had a fever of 38.2°C for 3 days, a persistent dry cough,
extreme fatigue, and I completely lost my sense of smell and taste yesterday.
I've been isolating at home.
```

### Possible UTI / Kidney Infection
```
I have pain and burning when I urinate, I need to go very frequently, 
and there's some lower back pain on my right side. I also have a mild fever of 37.9°C.
The symptoms started 2 days ago.
```

### Asthma-Like
```
I've been wheezing and having tightness in my chest for the past few hours.
I have a history of asthma but I don't have my inhaler. I can speak full sentences
but breathing feels labored when I exert myself.
```

---

## 🔴 HIGH RISK Scenarios (Emergency)

### Cardiac Emergency
```
I'm having severe crushing chest pain that radiates to my left arm and jaw.
I'm sweating profusely and feel nauseous. The pain started about 20 minutes ago
and isn't going away. I feel like something is very wrong.
```

### Stroke
```
My husband suddenly couldn't speak clearly — his words came out garbled.
His face is drooping on the right side and his right arm is very weak.
This happened about 10 minutes ago out of nowhere.
```

### Meningitis
```
My son has had a very high fever (39.8°C), severe headache, and his neck
is so stiff he can't touch his chin to his chest. He's very sensitive to light
and has been getting worse over the last 6 hours.
```

### Anaphylaxis
```
I ate peanuts about 5 minutes ago (I'm allergic). My throat feels like it's
swelling up, I'm having trouble breathing, and I have hives all over my body.
My lips feel tingly and swollen.
```

---

## 🔄 Multi-Turn Conversation Tests

### Turn 1 → Turn 2 Escalation
**Turn 1:**
```
I have a headache and some neck stiffness.
```

**Turn 2 (follow-up):**
```
The headache came on suddenly and is the worst headache I've ever had.
My neck is very stiff and I can't stand the light in the room.
I also have a fever of 39.5°C.
```
*Expected: System should escalate from possibly MODERATE to HIGH (meningitis red flags)*

---

### Symptom Clarification Flow
**Turn 1:**
```
I've been really tired lately and losing weight.
```

**Turn 2:**
```
I'm also very thirsty all the time and urinating a lot, especially at night.
My vision has been a bit blurry lately too.
```
*Expected: Diabetes mellitus should emerge as a strong possibility*

---

## 📊 Complex / Multi-System Presentations

### Possible Heart Failure
```
For the past month I've been increasingly short of breath, especially when I lie flat
to sleep — I now use 3 pillows. My ankles are very swollen. I've gained about 4kg
in the last 2 weeks and I'm exhausted climbing stairs. I'm 62 years old.
```

### Possible Hypothyroidism
```
Over the past few months I've been gaining weight despite not changing my diet,
feeling constantly cold when others are comfortable, my hair is falling out,
I'm always exhausted, and I've been feeling quite depressed. My bowel movements
have become very infrequent too.
```

### Possible DVT/PE
```
I flew long-haul last week (14 hours). Now my left calf is swollen, red, and
very painful. I also have some chest pain and shortness of breath that started today.
```
*Expected: HIGH risk — DVT/PE is a medical emergency*

---

## 📄 PDF Upload Test

Upload a PDF with content like:

```
Patient: John Doe, 45M
Chief Complaint: Chest pain and shortness of breath
BP: 145/95 mmHg
HR: 102 bpm
Temperature: 37.8°C
O2 Sat: 93%
History: Hypertension, smoker 20 pack-years
Current medications: Amlodipine 5mg daily
```

Then enter: "Please analyze this patient's report and assess their risk"

---

## ⚡ Edge Case Tests

### Vague Input
```
I don't feel well
```
*Expected: System should ask for clarification via follow-up questions*

### Single Critical Symptom
```
My chest really hurts
```
*Expected: HIGH risk, immediate emergency recommendation*

### Mental Health
```
I've been feeling extremely sad for months, lost interest in everything I used to enjoy,
can't sleep properly, and have been having thoughts that life isn't worth living.
```
*Expected: HIGH risk, mental health crisis resources recommended*

### Pediatric (add context)
```
My 18-month-old baby has had a fever of 39.2°C for the past 12 hours.
She's been crying a lot, not eating, and seems very lethargic. 
I noticed a small rash starting on her tummy.
```
*Expected: MODERATE-HIGH risk, urgent pediatric evaluation recommended*
