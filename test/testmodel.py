import unittest
import math
import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../src/")
import model

class TestModel(unittest.TestCase):
    def testARloglklh1(self):
        mod=model.AR(lag=1,phi=np.array([0.98]),sigma=1,intercept=2)
        x=np.zeros((1,2))
        x[0,0]=99
        x[0,1]=100
        l1=mod.loss(x)[0]
        phi=np.array([0.98])
        sigma=1
        intercept=2
        l2=-float(1)/2*math.log(2*math.pi)-float(1)/2*math.log(sigma**2)-float(1)/2/(sigma**2)*(x[0,1]-2-phi[0]*x[0,0])**2
        self.assertEqual(l1, l2)

    def testARlklh2(self):
        mod=model.AR(lag=2,phi=np.array([[1.01],[1.02]]),sigma=1,intercept=2)
        x=np.zeros((1,3))
        x[0,0]=98
        x[0,1]=99
        x[0,2]=100
        l1=mod.loss(x)[0]
        phi=np.array([[1.01],[1.02]])
        sigma=1
        intercept=2
        l2=-(intercept + np.dot(x, np.vstack((np.flipud(phi),np.array([-1.0])))))**2/2/(sigma**2)-math.log(sigma**2)/2-math.log(2*math.pi)/2
        self.assertEqual(l1,l2[0])

    def testARgrad1(self):
        mod=model.AR(lag=1,phi=np.array([0.98]),sigma=1,intercept=2)
        x=np.zeros((1,2))
        x[0,0]=99
        x[0,1]=100
        g1=mod.loss(x)[1]
        phi=np.array([0.98])
        sigma=1
        T=2
        intercept=2
        temp = intercept + np.dot(x, np.vstack((np.flipud(phi),-1.0)))
        grad_phi = np.zeros((1,1))-float(temp) * (np.fliplr(np.matrix(x[0,0]))).T
        grad_intercept = -float(temp)
        grad_sigma = float(temp)**2

        grad_phi = grad_phi / (sigma**2)
        grad_intercept = grad_intercept / (sigma**2)
        grad_sigma = grad_sigma / (sigma**3)
        grad_sigma -= float(1)/ (sigma)

        g2 = {} 
        g2['phi'] = grad_phi   
        g2['intercept'] = grad_intercept 
        g2['sigma'] = grad_sigma

        self.assertAlmostEqual(g1,g2)


    def testARgrad2(self):
        mod=model.AR(lag=2,phi=np.array([[1.01],[1.02]]),sigma=1,intercept=2)
        x=np.zeros((1,3))
        x[0,0]=98
        x[0,1]=99
        x[0,2]=100
        g1=mod.loss(x)[1]
        phi=np.array([[1.01],[1.02]])
        sigma=1
        T=3
        intercept=2
        temp = intercept + np.dot(x, np.vstack((np.flipud(phi),-1.0)))
        grad_phi = np.zeros((2,1))-float(temp) * (np.fliplr(np.matrix(x[0,0:2]))).T
        grad_intercept = -float(temp)
        grad_sigma = float(temp)**2

        grad_phi = grad_phi / (sigma**2)
        grad_intercept = grad_intercept / (sigma**2)
        grad_sigma = grad_sigma / (sigma**3)
        grad_sigma -= float(1)/ (sigma)

        g2 = {} 
        g2['phi'] = grad_phi   
        g2['intercept'] = grad_intercept 
        g2['sigma'] = grad_sigma

        self.assertAlmostEqual(g1['phi'].all(),g2['phi'].all())
        self.assertAlmostEqual(g1['intercept'],g2['intercept'])
        self.assertAlmostEqual(g1['sigma'],g2['sigma'])

    def testMAloglklh1(self):
        mod=model.MA(lag=1,phi=np.array([0.98]),sigma=1,intercept=2)
        x=np.zeros((1,2))
        x[0,0]=99
        x[0,1]=100
        l1=mod.loss(x)
        phi=np.array([0.98])
        sigma=1
        T=2
        intercept=2

        l2=-math.log(2*math.pi*sigma**2)

        """Derive autocorrelation for likelihood function"""
        autocov = np.zeros((2,1))
        autocov[0]=sigma**2+phi[0]**2*sigma**2
        autocov[1]=-phi[0]*sigma**2

        """Derive the covariance matrix for likelihood function"""
        covmat=np.zeros((2,2))
        covmat[0,0]=autocov[0]
        covmat[1,1]=autocov[0]
        covmat[0,1]=autocov[1]
        covmat[1,0]=autocov[1]
        
        l2 -= 0.5*math.log(abs(np.linalg.det(covmat)))+float(1)/2/sigma/sigma*np.matmul(np.matmul(x,np.linalg.inv(covmat)),np.transpose(x))[0,0]

        self.assertEqual(l1, l2)

    def testMAloglklh2(self):
        mod=model.MA(lag=2,phi=np.array([[1.01],[1.02]]),sigma=1,intercept=2)
        x=np.zeros((1,3))
        x[0,0]=98
        x[0,1]=99
        x[0,2]=100
        l1=mod.loss(x)
        phi=np.array([[1.01],[1.02]])
        sigma=1
        T=3
        intercept=2

        l2=-float(3)/2*math.log(2*math.pi*sigma**2)

        """Derive autocorrelation for likelihood function"""
        autocov = np.zeros((3,1))
        autocov[0]=sigma**2+np.dot(phi,phi)*sigma**2[0,0]
        for i in range(2):
                autocov[i+1]=np.dot(phi[0:lag-i-2],phi[i+1:lag-1])*sigma**2[0,0]-phi[i]*sigma**2

        """Derive the covariance matrix for likelihood function"""
        covmat=np.zeros((3,3))
        for i in range(3):
            for j in range(i+1):
                if abs(i-j)<=2:
                    covmat[i,j]=autocov[abs(i-j)]
                    covmat[j,i]=autocov[abs(i-j)]
        
        l2 -= 0.5*math.log(abs(np.linalg.det(covmat)))+float(1)/2/sigma/sigma*np.matmul(np.matmul(np.transpose(x),np.linalg.inv(covmat)),x)[0,0]
        self.assertEqual(l1,l2)




if __name__ == "__main__":
    unittest.main()



