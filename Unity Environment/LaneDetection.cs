using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Linq;
using Unity.MLAgents.Sensors;
using UnityEngine;


public class LaneDetection : MonoBehaviour
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
    float dis_left = 0;
    float dis_right = 0;

    public void Awake() // ���Ǽҵ尡 ���۵Ǳ� ���� ��� ���� �ʱ�ȭ�ϴ� �޼ҵ� (start���� ���� ȣ���)
    {
        tr = GetComponent<Transform>();
        rb = GetComponent<Rigidbody>();
        startPosition = tr.position;
        startRotation = tr.eulerAngles;
        dis_Finish = 55f;       // ���⼭ ���� �ʱ�ȭ�� ������� FixedUpdate() �����

        // �������� �ʱ�ȭ
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // ��ġ �ʱ�ȭ
        tr.position = startPosition;
        tr.eulerAngles = startRotation;

        

    }

    private void FixedUpdate() // ���� ������Ʈ�� ���õ� �۾� ���� �޼ҵ� 
    {
        var raySensorComponent = GetComponent<RayPerceptionSensorComponent3D>();    // �ش� ������Ʈ�� raySensorComponent�� ����
        var input = raySensorComponent.GetRayPerceptionInput(); // ray sensor�� ��� �����͸� input�� ����
        var output = RayPerceptionSensor.Perceive(input); // ray cast (���� ����) ����� output�� ���� 
        step++;
        //Debug.Log("step"+step);
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
                else if (rayIndex == 2)
                {
                    dis_left = Vector3.Distance(hitposition, startPositionWorld);
                    //Debug.Log("dis_left : " + Vector3.Distance(hitposition, startPositionWorld));

                }
               
            }
            
        }

        float steerRatio = dis_left - dis_right; //  ����:2.985535 ��:2.185532
        //Debug.Log("steerRatio : " + steerRatio);

        if (steerRatio <= 3.58f && steerRatio >= 2.97f) // 1���� ���θ� �����ϴ�
        {
            Turn = 0;
            moveSpeed = 10f;

        }
        else if (steerRatio < 2.97f && steerRatio > 0f) // �߾Ӽ������� �̵��� ��
        {
            Turn = Mathf.Clamp(dis_left - dis_right + 0.05f, -1.0f, 1.0f);
            moveSpeed = 5.5f;
        }
        else if (steerRatio < 10.85f && steerRatio > 3.58f) // ���������� �̵��� ��
        {
            Turn = Mathf.Clamp(dis_right - dis_left - 0.05f, -1.0f, 1.0f);
            moveSpeed = 5.5f;
        }
        else if (steerRatio < 11.19 && steerRatio >= 10.85f) //  �߾Ӽ� �Ѿ��
        {
            Turn = Mathf.Clamp(dis_left - dis_right , -1.0f, 1.0f);
            moveSpeed = 5.5f;
        }
        else if (steerRatio >= 11.19) // �����ʿ��� �־������� ��
        {
            Turn = Mathf.Clamp(dis_right - dis_left, -1.2f, 1.2f);
            moveSpeed = 5.5f;
        }
        else if (steerRatio <= 0f) // �߾Ӽ����� �־����� �� ��
        {
            Turn = Mathf.Clamp(dis_right - dis_left, -1.0f, 1.0f);
            moveSpeed = 5.5f;
        }
      

        tr.Translate(moveSpeed * Time.fixedDeltaTime * Vector3.forward); // Translate�� ��ü�� �̵���Ű�� �޼ҵ�
        tr.Rotate(new Vector3(0f, Turn, 0f));    // Rotate�� ��ü�� ȸ����Ű�� �޼ҵ� 

        dis_Traveled = Vector3.Distance(tr.position, startPosition);
        //Debug.Log("dis_Traveled : " + dis_Traveled);

        if (dis_Traveled >= dis_Finish)
        {
            //Debug.Log("Finish");

            tr.position = startPosition;
            tr.eulerAngles = startRotation;
            moveSpeed = 10f;
            dis_Traveled = 0f;
            step = 0;
        }


    }

    /*private void OnTriggerEnter(Collider other) // �浹 �� ����Ǵ� �޼ҵ� 
    {
        if (other.CompareTag("wall")) // ������ �浹 ��
        {
            // �������� �ʱ�ȭ
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;

            // ��ġ �ʱ�ȭ
            tr.position = startPosition;
            tr.eulerAngles = startRotation;

        }
    }*/

}
