import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, { useNodesState, useEdgesState, addEdge, MiniMap, Controls } from 'reactflow';
import 'reactflow/dist/style.css';
import AccountNode from './Accountode';
import CustomEdge from './CustomEdge'; // Adjust the path as per your file structure


const Graph = () => {
    const [accountNumber, setAccountNumber] = useState('');
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);
    const [showCanvas, setShowCanvas] = useState(false);

    const handleAccountNumberChange = (e) => {
        setAccountNumber(e.target.value);
    };

    const fetchData = useCallback(async () => {
        try {
            const myHeaders = new Headers();
            myHeaders.append('Content-Type', 'application/json');
            const raw = JSON.stringify({
                account_number: accountNumber,
            });
            const requestOptions = {
                method: 'POST',
                headers: myHeaders,
                body: raw,
                redirect: 'follow',
            };
            const response = await fetch('http://127.0.0.1:5000/transactions', requestOptions);
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const responseData = await response.json();

            const accounts = new Set();
            const extractAccounts = (transactions) => {
                if (transactions) {
                    transactions.forEach((transaction) => {
                        accounts.add(transaction.Account);
                        accounts.add(transaction['Account.1']);
                    });
                }
            };
            extractAccounts(responseData.transactions_from_account);
            extractAccounts(responseData.transactions_to_account);

            // Prepare nodes
            const initialNodes = Array.from(accounts).map((account, index) => ({
                id: account,
                data: { label: account },
                position: {
                    x: Math.cos((index / accounts.size) * 2 * Math.PI) * 400,
                    y: Math.sin((index / accounts.size) * 2 * Math.PI) * 400 - 300,
                },
                type: 'accountNode',
            }));

            // Add central node
            initialNodes.push({
                id: 'central',
                data: { label: accountNumber },
                position: { x: 0, y: 0 },
                type: 'accountNode',
            });

            // Prepare edges
            const edgesData = [];
            const existingEdges = new Set(); // To track existing edges
            const extractEdges = (transactions) => {
                transactions.forEach((transaction, i) => {
                    if (transaction.Account && transaction['Account.1']) {
                        const edgeId = `${transaction.Account}-${transaction['Account.1']}`;
                        console.log(`${transactions}`);
                        if (!existingEdges.has(edgeId)) {
                            const edgeColor = transaction['Is Laundering'] === 0 ? 'green' : 'red';
                            edgesData.push({
                                id: edgeId,
                                source: transaction.Account,
                                target: transaction['Account.1'],
                                label: transaction["Amount Paid"],
                                type: 'bezier',
                                animated: true,
                                style: {
                                  stroke: edgeColor,
                                  fontSize: '16px', 
                                  lineHeight: '1.5', 
                                },
                              });
                              existingEdges.add(edgeId);
                        }
                    }
                });
            };
            

            extractEdges(responseData.transactions_from_account);
            extractEdges(responseData.transactions_to_account);

            setNodes(initialNodes);
            setEdges(edgesData);
            setShowCanvas(true);
        } catch (error) {
            console.error('Error fetching data:', error);
            setNodes([]);
            setEdges([]);
        }
    }, [accountNumber]);

    const onConnect = useCallback(
        (params) => {
            const edgeId = `${params.source}-${params.target}`;
            if (!edges.find((edge) => edge.id === edgeId)) {
                setEdges((eds) => addEdge({ ...params, id: edgeId, animated: true }, eds));
            }
        },
        [edges]
    );

    const nodeTypes = { accountNode: AccountNode };

    const edgeTypes = {
        custom: CustomEdge,
    };

    return (
        <div className="h-[100vh] w-full flex items-center justify-center">
            {!showCanvas ? (
                <div className="flex flex-col items-center">
                    <input
                        type="text"
                        className="border border-gray-300 px-3 py-2 rounded-md focus:outline-none focus:ring focus:border-blue-500 mb-4"
                        placeholder="Enter account number"
                        value={accountNumber}
                        onChange={handleAccountNumberChange}
                    />
                    <button
                        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                        onClick={fetchData}
                    >
                        Fetch Transactions
                    </button>
                </div>
            ) : (
                <ReactFlow
                    nodes={nodes}
                    edges={edges}
                    onNodesChange={onNodesChange}
                    onEdgesChange={onEdgesChange}
                    onConnect={onConnect}
                    nodeTypes={nodeTypes}
                    edgeTypes={edgeTypes}
                    snapToGrid
                    snapGrid={[20, 20]}
                    defaultViewport={{ x: 0, y: 0, zoom: 1.5 }}
                    fitView
                    attributionPosition="bottom-left"
                >
                    <MiniMap
                        nodeStrokeColor={(n) => {
                            if (n.type === 'accountNode') return '#0041d0';
                        }}
                        nodeColor={(n) => {
                            if (n.type === 'accountNode') return '#fff';
                            return '#000';
                        }}
                    />
                    <Controls />
                </ReactFlow>
            )}
        </div>
    );
};

export default Graph;
