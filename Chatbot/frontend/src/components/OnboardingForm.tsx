// src/components/OnboardingForm.tsx
import { FC, useState } from 'react';
import { OnboardingData } from '../types';
import { Check } from 'lucide-react';

interface OnboardingFormProps {
  onComplete: (data: OnboardingData) => void;
  onStepChange: (step: number) => void;
}

const OnboardingForm: FC<OnboardingFormProps> = ({ onComplete, onStepChange }) => {
  const [step, setStep] = useState(0);
  const [formData, setFormData] = useState<OnboardingData>({
    fullName: '',
    inverterModel: '',
    serialNumber: '',
    installedDate: '',
    country: '',
    pinCode: '',
    address: '',
  });
  const [error, setError] = useState('');

  const steps = [
    { label: 'Full Name', key: 'fullName', placeholder: 'Enter your full name', type: 'text' },
    { label: 'Inverter Model', key: 'inverterModel', placeholder: 'Enter inverter model', type: 'text' },
    { label: 'Serial Number', key: 'serialNumber', placeholder: 'Enter serial number', type: 'text' },
    { label: 'Installed Date', key: 'installedDate', placeholder: 'YYYY-MM-DD', type: 'date' },
    { label: 'Country', key: 'country', placeholder: 'Enter your country', type: 'text' },
    { label: 'Pin Code', key: 'pinCode', placeholder: 'Enter your PIN code', type: 'text' },
    { label: 'Address', key: 'address', placeholder: 'Enter your full address', type: 'textarea' },
  ];

  const handleNext = (e: React.FormEvent) => {
    e.preventDefault();
    const currentKey = steps[step].key as keyof OnboardingData;
    const value = formData[currentKey];

    if (!value.trim()) {
      setError(`${steps[step].label} is required.`);
      return;
    }

    if (currentKey === 'installedDate') {
      const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
      if (!dateRegex.test(value)) {
        setError('Please enter a valid date in YYYY-MM-DD format.');
        return;
      }
    }

    if (currentKey === 'pinCode' && !/^\d{5,6}$/.test(value)) {
      setError('Please enter a valid 5 or 6-digit PIN code.');
      return;
    }

    setError('');
    if (step < steps.length - 1) {
      setStep(step + 1);
      onStepChange(step + 1);
    } else {
      onComplete(formData);
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-md bg-white rounded-3xl shadow-2xl p-8">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold text-gray-800">Setup Your Inverter</h2>
          <p className="text-gray-600 text-sm">
            Step {step + 1} of {steps.length}
          </p>
        </div>
        <form onSubmit={handleNext} className="space-y-4">
          <div>
            <label
              htmlFor={steps[step].key}
              className="block text-sm font-medium text-gray-700"
            >
              {steps[step].label}
            </label>
            {steps[step].type === 'textarea' ? (
              <textarea
                id={steps[step].key}
                name={steps[step].key}
                value={formData[steps[step].key as keyof OnboardingData]}
                onChange={handleChange}
                placeholder={steps[step].placeholder}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                rows={4}
                aria-required="true"
              />
            ) : (
              <input
                id={steps[step].key}
                name={steps[step].key}
                type={steps[step].type}
                value={formData[steps[step].key as keyof OnboardingData]}
                onChange={handleChange}
                placeholder={steps[step].placeholder}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                aria-required="true"
              />
            )}
            {error && (
              <p className="text-red-500 text-xs mt-1">{error}</p>
            )}
          </div>
          <div className="flex gap-2">
            {step > 0 && (
              <button
                type="button"
                onClick={() => {
                  setStep(step - 1);
                  onStepChange(step - 1);
                  setError('');
                }}
                className="flex-1 bg-gray-300 text-gray-800 py-2 rounded-lg hover:bg-gray-400"
              >
                Back
              </button>
            )}
            <button
              type="submit"
              className="flex-1 bg-blue-500 text-white py-2 rounded-lg hover:bg-blue-600 flex items-center justify-center gap-2"
            >
              <Check size={20} />
              {step === steps.length - 1 ? 'Finish' : 'Next'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default OnboardingForm;