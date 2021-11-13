// import * as React from "react";
import React, { useState } from "react";
import axios from "axios";
import TextField from "@mui/material/TextField";
import Container from "@mui/material/Container";

function App() {
  const [prompt, setPrompt] = useState("Haizhi");

  const fetchData = async () => {
    const result = await axios(
      "http://127.0.0.1:8000/evaluate-prompt/" + prompt
    );
    console.log("returned result: ", result);
    // setData(result.data);
  };

  const handleChange = (event) => {
    setPrompt(event.target.value);
  };
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      console.log("prompt entered: ", prompt);
      fetchData();
    }
  };
  return (
    <Container maxWidth="sm" sx={{ marginTop: "5rem" }}>
      <TextField
        fullWidth
        label="prompt"
        variant="outlined"
        onChange={handleChange}
        onKeyPress={handleKeyPress}
        defaultValue={prompt}
      />
    </Container>
  );
}

export default App;
