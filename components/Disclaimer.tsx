import React from 'react';

interface DisclaimerProps {
  children: React.ReactNode;
}

const Disclaimer: React.FC<DisclaimerProps> = ({ children }) => {
  return (
    <div className="bg-yellow-50 border-2 border-warning text-yellow-800 p-4 rounded-lg mt-5">
      <strong className="text-amber-700">⚠️ Important:</strong> {children}
    </div>
  );
};

export default Disclaimer;