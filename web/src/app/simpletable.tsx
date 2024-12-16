import React from 'react';

interface TableProps {
    headers: string[];
    data: { [key: string]: string }[];
}

const SimpleTable: React.FC<TableProps> = ({ 
    headers = ["Column 1", "Column 2"], 
data = Array.from({ length: 5 }, () => ({
    "Column 1": `Sample Data ${Math.floor(Math.random() * 100)}`,
    "Column 2": `Sample Data ${Math.floor(Math.random() * 100)}`
}))
}) => {
    return (
        <table className="simple-table">
            <thead>
                <tr>
                    {headers.map((header, index) => (
                        <th key={index}>{header}</th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {data.length > 0 ? (
                    data.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                            {headers.map((header, colIndex) => (
                                <td key={colIndex}>{row[header]}</td>
                            ))}
                        </tr>
                    ))
                ) : (
                    <tr>
                        <td colSpan={headers.length}>No data available.</td>
                    </tr>
                )}
            </tbody>
        </table>
    );
};

export default SimpleTable;