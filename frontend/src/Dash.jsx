import React, { useEffect, useState } from "react";

const Dash = () => {
	const totalTransactions = 1000;
	const suspiciousTransactions = 50;
	const totalBitcoinInvolved = 250;
	const suspiciousPercentage =
		(suspiciousTransactions / totalTransactions) * 100;

	const [animationClass, setAnimationClass] = useState("");

	useEffect(() => {
		// Set animation class with delay for each div
		setTimeout(() => setAnimationClass("fadeIn"), 500);
		setTimeout(() => setAnimationClass("fadeIn"), 1000);
		setTimeout(() => setAnimationClass("fadeIn"), 1500);
		setTimeout(() => setAnimationClass("fadeIn"), 2000);
	}, []);

	return (
		<div className="flex  w-full">
			<div className={`flex-1 m-2 animated ${animationClass}`}>
				<div className="rounded-lg bg-gray-300 p-4">
					<h2>
						Total number of transactions analyzed:{" "}
						{totalTransactions}
					</h2>
				</div>
			</div>
			<div className={`flex-1 m-2 animated ${animationClass}`}>
				<div className="rounded-lg bg-gray-300 p-4">
					<h2>
						Total number of suspicious transactions detected:{" "}
						{suspiciousTransactions}
					</h2>
				</div>
			</div>
			<div className={`flex-1 m-2 animated ${animationClass}`}>
				<div className="rounded-lg bg-gray-300 p-4">
					<h2>
						Total amount of Bitcoin involved in suspicious
						transactions: {totalBitcoinInvolved} BTC
					</h2>
				</div>
			</div>
			<div className={`flex-1 m-2 animated ${animationClass}`}>
				<div className="rounded-lg bg-gray-300 p-4">
					<h2>
						Percentage of suspicious transactions compared to total
						transactions: {suspiciousPercentage.toFixed(2)}%
					</h2>
				</div>
			</div>
		</div>
	);
};

export default Dash;
