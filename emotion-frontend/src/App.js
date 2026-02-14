import { useState } from "react";
import axios from 'axios'


function App() {
  const [image , setImage] = useState()
  const [prev , setPrev] = useState()
  const [emotion , setEmotion] = useState()

  const handleImageChange = (e)=>{
    const file = e.target.files[0]
    console.log(file)
    setImage(file);
    setPrev(URL.createObjectURL(file))
  };

  const handleSubmit = async ()=>{
    if(!image) return alert('Please upload an image')
    
    const formData = new FormData();
    formData.append("file",image);
     try {
      const res = await axios.post("http://127.0.0.1:8000/predict_emotion", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      console.log(res.data)
      setEmotion(res.data.emotion);
    } catch (error) {
      console.error(error);
      alert("Error connecting to backend");
    }
  }

  return (
     <div className="container">
      <h1>Emotion Detection</h1>

      <input type="file" accept="image/*" onChange={handleImageChange} />

      {prev && (
        <div className="prev">
          <img src={prev} alt="Uploaded" />
        </div>
      )}

      <button onClick={handleSubmit}>Predict Emotion</button>

      {emotion && <h2>Predicted Emotion: {emotion}</h2>}
    </div>
  );
}

export default App;
