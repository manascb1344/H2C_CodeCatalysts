import React, { useState } from 'react';

const Graph = () => {
    const [number1, setNumber1] = useState('');
    const [number2, setNumber2] = useState('');

    const handleNumber1Change = (e) => {
        const value = e.target.value;
        if (!isNaN(value)) {
            setNumber1(value);
        }
    };

    const handleNumber2Change = (e) => {
        const value = e.target.value;
        if (!isNaN(value)) {
            setNumber2(value);
        }
    };

    return (
        <div className="flex space-x-4">
            <input
                type="number"
                className="border border-gray-300 px-3 py-2 rounded-md focus:outline-none focus:ring focus:border-blue-500"
                placeholder="Enter number 1"
                value={number1}
                onChange={handleNumber1Change}
            />
            <input
                type="number"
                className="border border-gray-300 px-3 py-2 rounded-md focus:outline-none focus:ring focus:border-blue-500"
                placeholder="Enter number 2"
                value={number2}
                onChange={handleNumber2Change}
            />
        </div>
    );
};

export default Graph;
