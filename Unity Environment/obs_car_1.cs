using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Linq;
using Unity.MLAgents.Sensors;
// using UnityEditor.PackageManager;
using UnityEngine;

public class obs_car_1 : MonoBehaviour
{
    private Transform tr;
    private Rigidbody rb;
    private float moveSpeed = 10f;
    private float Turn = 0f;
    private Vector3 direction;

    public float dis_Finish; // ���� �Ÿ�
    private float dis_Traveled;
    Vector3 startPosition;
    Vector3 startRotation;

    int step = 0;
    //float dis_left = 0;
    float dis_right = 0;
    float targetDisRight = 1.17341f;
    float tolernace = 0.15f;

    float Kp = 3.0f; // ��� ���� ���
    float Ki = 1.5f; // ���� ���� ���
    float Kd = 3.0f; // �̺� ���� ���

    float integral = 0f; // ���� ��
    float previousError = 0f; // ���� ���� ��

    private RayPerceptionSensorComponent3D raySensorComponent;

    float error;
    float proportional;
    float derivative;
    float pidValue;

    public void Awake() // ���Ǽҵ尡 ���۵Ǳ� ���� ��� ���� �ʱ�ȭ�ϴ� �޼ҵ� (start���� ���� ȣ���)
    {
        tr = GetComponent<Transform>();
        rb = GetComponent<Rigidbody>();
        startPosition = tr.position;
        startRotation = tr.eulerAngles;
        //dis_Finish = 400f;       // ���⼭ ���� �ʱ�ȭ�� ������� FixedUpdate() �����

        // �������� �ʱ�ȭ
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // ��ġ �ʱ�ȭ
        tr.position = startPosition;
        tr.eulerAngles = startRotation;

        raySensorComponent = GetComponent<RayPerceptionSensorComponent3D>();    // �ش� ������Ʈ�� raySensorComponent�� ����

    }

    private void FixedUpdate() // ���� ������Ʈ�� ���õ� �۾� ���� �޼ҵ� 
    {

        var input = raySensorComponent.GetRayPerceptionInput(); // ray sensor�� ��� �����͸� input�� ����
        var output = RayPerceptionSensor.Perceive(input); // ray cast (���� ����) ����� output�� ���� 
        step++;
        //Debug.Log("step" + step);
        for (var rayIndex = 0; rayIndex < output.RayOutputs.Length; rayIndex++)
        {
            var extent = input.RayExtents(rayIndex); // �־��� rayindex�� ���� ����/���� ����Ʈ�� extent�� ���� 
            Vector3 startPositionWorld = extent.StartPositionWorld; // rayindex�� ���� ����Ʈ�� �ش� ������ ����
            Vector3 endPositionWorld = extent.EndPositionWorld; // rayindex�� ���� ����Ʈ�� �ش� ������ ����

            var rayOutput = output.RayOutputs[rayIndex];

            if (rayOutput.HasHit) // ray�� �ε����� �� �Ÿ� ���
            {
                // Lerp()�� spw + (epw-spw)*ht �̿��� ���� (�� �� ������ ���� ���� ��� ���� ������ ã�µ� ���)
                Vector3 hitposition = Vector3.Lerp(startPositionWorld, endPositionWorld, rayOutput.HitFraction);    // hitFraction�� ���� ��ü������ ����ȭ�� �Ÿ� 
                //Debug.DrawLine(startPositionWorld, hitposition, Color.red);

                if (rayIndex == 1)
                {
                    dis_right = Vector3.Distance(hitposition, startPositionWorld); // Vector3.Distance�� ���ڰ� �Ÿ��� ��ȯ
                    //Debug.Log("dis_right :  " + Vector3.Distance(hitposition, startPositionWorld));
                }

            }

        }

        error = targetDisRight - dis_right;

        // PID ���� ���
        proportional = error;
        integral += error * Time.deltaTime;
        derivative = (error - previousError) / Time.deltaTime;

        pidValue = Kp * proportional + Ki * integral + Kd * derivative;

        if (Mathf.Abs(pidValue) < tolernace)
        {
            Turn = 0;
            moveSpeed = 10.0f;
        }
        else
        {
            Turn = Mathf.Clamp(-pidValue, -30f, 30f);
            moveSpeed = 5.5f;
        }

        tr.Translate(moveSpeed * Time.fixedDeltaTime * Vector3.forward); // Translate�� ��ü�� �̵���Ű�� �޼ҵ�
        tr.Rotate(new Vector3(0f, Turn, 0f));    // Rotate�� ��ü�� ȸ����Ű�� �޼ҵ� 

        dis_Traveled = Vector3.Distance(tr.position, startPosition);
        //Debug.Log("dis_Traveled : " + dis_Traveled);

    }

    private void OnTriggerEnter(Collider other) // �浹 �� ����Ǵ� �޼ҵ� 
    {
        if (other.CompareTag("obstacle")) // ��ֹ� ������ �浹 ��
        {
            // �������� �ʱ�ȭ
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;

            // ��ġ �ʱ�ȭ
            tr.position = startPosition;
            tr.eulerAngles = startRotation;

        }
    }

}
