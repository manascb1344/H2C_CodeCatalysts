import React, { useEffect, useState } from "react"; // You can create a CSS file to style your component

const Dash = () => {
  const totalTransactions = 1000;
  const suspiciousTransactions = 50;
  const totalBitcoinInvolved = 250;
  const suspiciousPercentage = (suspiciousTransactions / totalTransactions) * 100;

  const [animationClass, setAnimationClass] = useState("");

  useEffect(() => {
    // Set animation class with delay for each div
    setTimeout(() => setAnimationClass("fadeIn"), 500);
    setTimeout(() => setAnimationClass("fadeIn"), 1000);
    setTimeout(() => setAnimationClass("fadeIn"), 1500);
    setTimeout(() => setAnimationClass("fadeIn"), 2000);
  }, []);

  return (
    <div className="grid grid-rows-2 grid-cols-2 h-full w-full">
      <div
        className={`animated ${animationClass} shadow-2xl border-2 border-white bg-gradient-to-br from-blue-400 to-blue-300 p-4 m-2 flex items-center justify-center rounded-lg`}
      >
        <h2 className="text-white text-3xl font-bold">
          Total transactions analyzed: {totalTransactions}
        </h2>
      </div>
      <div
        className={`animated ${animationClass} shadow-2xl  border-2 border-white bg-gradient-to-br bg-green-500 p-4 m-2 flex items-center justify-center rounded-lg`}
      >
        <h2 className="text-white text-3xl font-bold">
          Suspicious transactions detected: {suspiciousTransactions}
        </h2>
      </div>
      <div
        className={`animated ${animationClass} shadow-2xl  border-2 border-white bg-gradient-to-br from-cyan-500 to-cyan-300 p-4 m-2 flex items-center justify-center rounded-lg`}
      >
        <h2 className="text-white text-3xl font-bold">
          Bitcoin involved in suspicious transactions: {totalBitcoinInvolved} BTC
        </h2>
      </div>
      <div
        className={`animated ${animationClass} shadow-2xl  bg-gradient-to-br from-teal-500 to-teal-300 p-4 m-2 flex items-center justify-center rounded-lg`}
      >
        <h2 className="text-white text-3xl font-bold">
          Suspicious transactions percentage: {suspiciousPercentage.toFixed(2)}%
        </h2>
      </div>
    </div>
  );
};

export default Dash;
