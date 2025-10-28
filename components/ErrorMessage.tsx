import React from 'react';

interface ErrorMessageProps {
  message: string;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ message }) => {
  return (
    <div className="bg-red-50 border-2 border-error text-red-800 p-4 rounded-lg mt-4">
      <span className="font-semibold">❌ Error:</span> {message}
    </div>
  );
};

export default ErrorMessage;